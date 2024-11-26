import asyncio
from typing import Optional, List
from pathlib import Path
from nonebot import logger
from nonebot.adapters.onebot.v11 import MessageEvent, MessageSegment, Bot, Message
from nonebot_plugin_alconna import Alconna, on_alconna
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import httpx
from io import BytesIO

FONT_PATH = Path("YaHei Consolas Hybrid 1.12.ttf")
SUB_FONT_PATH = Path("YaHei Consolas Hybrid 1.12.ttf")

cmd = Alconna(".q")
command = on_alconna(cmd, aliases={"q"})

@command.handle()
async def handle_command(bot: Bot, event: MessageEvent):
    if not event.reply:
        await command.send(MessageSegment.text("请回复一条消息以使用此命令。"))
        return

    reply_content = extract_reply_content(event.reply.message)
    user_id = event.reply.sender.user_id
    nickname = event.reply.sender.nickname
    if user_id == event.self_id:
        await command.send(MessageSegment.text("Σ( ° △ °|||)︴\n不能回复咱自己的信息"))
        return

    avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
    avatar_image = await fetch_image_from_url(avatar_url)

    if avatar_image is None:
        await command.send(MessageSegment.text("无法获取用户头像，请检查网络连接。"))
        return

    image_segment = next((seg for seg in event.reply.message if seg.type == "image"), None)
    if image_segment:
        image_url = image_segment.data['url']
        additional_image = await fetch_image_from_url(image_url)

    result_image = await process_image(avatar_image, reply_content, nickname, additional_image)

    if result_image is not None:
        image_bytes = BytesIO()
        result_image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        await command.send(MessageSegment.image(image_bytes))
    else:
        await command.send(MessageSegment.text("处理图片时发生错误，请检查输入参数。"))

def extract_reply_content(reply_msg: Message) -> str:
    reply_content = ""
    for segment in reply_msg:
        if segment.type == "text":
            reply_content += segment.data['text']
    return reply_content.strip()

async def fetch_image_from_url(url: str) -> Optional[Image.Image]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            image_data = response.content
            return Image.open(BytesIO(image_data))
        except httpx.RequestError as e:
            logger.warning(f"Failed to fetch image from {url}: {e}")
            return None

async def process_image(avatar_image: Image.Image, text: str, nickname: str, additional_image: Optional[Image.Image] = None) -> Optional[Image.Image]:
    loop = asyncio.get_event_loop()

    def _process():
        png_path = "base3.png"  # base3
        left_margin = 482  #   730
        right_margin = 960
        bottom_margin = 2
        png_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if png_image is None:
            logger.warning("PNG Image not found, please check the path.")
            return None

        png_height, png_width, _ = png_image.shape
        target_width, target_height = 402, 402
        start_X, start_Y = 40, 40

        avatar_resized = cv2.resize(cv2.cvtColor(np.array(avatar_image), cv2.COLOR_RGBA2BGRA), (target_width, target_height))

        background = np.zeros((png_height, png_width, 4), dtype=np.uint8)
        background[:, :, :3] = (173, 69, 63)[::-1]
        background[:, :, 3] = 255

        background[start_X:(target_width + start_X), start_Y:(target_height + start_Y), :3] = avatar_resized[:, :, :3]

        alpha_channel = png_image[:, :, 3].astype(float) / 255
        background_float = background.astype(float)

        for channel in range(3):
            mask = alpha_channel > 0
            background_float[mask, channel] = (alpha_channel[mask] * png_image[mask, channel] +
                                               (1 - alpha_channel[mask]) * background_float[mask, channel])

        result = background_float.astype(np.uint8)

        if additional_image:
            additionalimage = additional_image.convert("RGBA")
            additional_np_array = np.array(additionalimage)
            img_height, img_width, _ = additional_np_array.shape

            max_img_width = (right_margin - left_margin) * 2 // 3
            min_img_width = (right_margin - left_margin) // 3

            max_img_height = png_height * 3 // 4
            max_img_width = (right_margin - left_margin) * 2 // 3
            min_img_width = (right_margin - left_margin) // 3

            scale_factor_height = max_img_height / img_height
            scale_factor_width = max_img_width / img_width
            scale_factor = min(scale_factor_height, scale_factor_width)

            if img_width < min_img_width:
                scale_factor = max(scale_factor, min_img_width / img_width)

            resized_additional_image = cv2.resize(cv2.cvtColor(additional_np_array, cv2.COLOR_RGBA2BGRA), (int(img_width * scale_factor), int(img_height * scale_factor)))

            resized_img_height, resized_img_width, _ = resized_additional_image.shape

            paste_x = left_margin + (right_margin - left_margin - resized_img_width) // 2
            paste_y = png_height // 2 - resized_img_height // 2

            if paste_y + resized_img_height < png_height and paste_x + resized_img_width < png_width:
                alpha_channel_add = resized_additional_image[:, :, 3].astype(float) / 255
                for channel in range(3):
                    mask = alpha_channel_add > 0
                    masked_alpha = alpha_channel_add[mask]
                    masked_resized_image = resized_additional_image[mask, channel]
                    masked_result = result[paste_y:paste_y+resized_img_height, paste_x:paste_x+resized_img_width, channel][mask]
                    result[paste_y:paste_y+resized_img_height, paste_x:paste_x+resized_img_width, channel][mask] = (
                        masked_alpha * masked_resized_image +
                        (1 - masked_alpha) * masked_result
                    )


        result = add_text_at_center(result, text, nickname, FONT_PATH, SUB_FONT_PATH, avatar_resized, left_margin, right_margin, bottom_margin)

        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))

    return await loop.run_in_executor(None, _process)


def calculate_average_color(image: np.ndarray) -> tuple:
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    bgr_mean = average_color[:3]
    return tuple(int(c) for c in bgr_mean)


def add_text_at_center(image: np.ndarray, text: str, nickname: str, font_path: Path, sub_font_path: Path, avatar_image: np.ndarray, left_margin: int, right_margin: int, bottom_margin: int) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), 42)
    sub_font = ImageFont.truetype(str(sub_font_path), 28)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(pil_image)

    avg_color = calculate_average_color(avatar_image)
    avg_color_rgb = (avg_color[2], avg_color[1], avg_color[0])

    text_lines = split_text_to_fit(draw, text, font, right_margin - left_margin)

    total_height = sum([draw.textbbox((0, 0), line, font=font)[3] for line in text_lines])
    total_height += len(text_lines) * 10

    base3_height = pil_image.height
    vertical_center = base3_height // 2
    current_y = vertical_center - total_height // 2

    for line in text_lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = left_margin + (right_margin - left_margin - text_width) // 2
        draw.text((text_x, current_y), line, fill=avg_color_rgb, font=font)
        current_y += text_height + 10

    sub_text = f"——{nickname}"
    sub_text_lines = split_text_to_fit(draw, sub_text, sub_font, right_margin - left_margin)

    sub_total_height = sum([draw.textbbox((0, 0), line, font=sub_font)[3] for line in sub_text_lines])
    sub_total_height += len(sub_text_lines) * 10

    sub_current_y = pil_image.height - bottom_margin - sub_total_height

    for sub_line in sub_text_lines:
        sub_text_bbox = draw.textbbox((0, 0), sub_line, font=sub_font)
        sub_text_width = sub_text_bbox[2] - sub_text_bbox[0]
        sub_text_height = sub_text_bbox[3] - sub_text_bbox[1]
        sub_text_x = left_margin + (right_margin - left_margin - sub_text_width) // 2
        draw.text((sub_text_x, sub_current_y), sub_line, fill=(180, 180, 180), font=sub_font)
        sub_current_y += sub_text_height + 10

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)

def split_text_to_fit(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines = []
    current_line = ""

    for char in text:
        test_line = f"{current_line}{char}"
        test_bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = test_bbox[2] - test_bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return lines






