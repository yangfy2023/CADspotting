import numpy as np
import json
from lxml import etree
import os
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from cairosvg import svg2png
from PIL import Image, ImageDraw, ImageFont
import pickle
from tqdm import tqdm
import argparse

SVG_CATEGORIES = [
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
]


def get_color_for_label(label):
    return tuple(SVG_CATEGORIES[label]["color"])


def get_name_for_label(label):
    return SVG_CATEGORIES[label]["name"]


def get_id_for_label(label):
    return SVG_CATEGORIES[label]["id"]


def draw_bboxes(
    image_path, bboxes, output_path, colors=None, widths=None, labels=None, alpha=64, scores=None, font_size=14, scale=1
):
    """
    在图像上绘制多个边界框并保存图像。

    参数:
    image_path (str): 图像文件路径。
    bboxes (list of tuples): 边界框列表，每个边界框格式为 (x1, y1, x2, y2)。
    output_path (str): 保存图像的路径。
    colors (list of str): 边界框的颜色列表，默认为 None(使用红色）。
    widths (list of int): 边界框的线宽列表，默认为 None(使用 2)。
    labels (list of str): 边界框的标签列表，默认为 None。
    alpha (int): 边界框的透明度，范围为 0（完全透明）到 255（完全不透明），默认为 128。
    """
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    if colors is None:
        colors = (255, 255, 255) * len(bboxes)
    if widths is None:
        widths = [2] * len(bboxes)
    if labels is None:
        labels = [None] * len(bboxes)

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # 尝试使用 Arial 字体
    except IOError:
        font = ImageFont.load_default(font_size)
    for bbox, color, width, label, score in zip(bboxes, colors, widths, labels, scores):
        # if label in range(30, 36):
        #     continue
        if label not in [0,1,2,3,4,5]:
            continue
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        draw_overlay.rectangle([x1, y1, x2, y2], outline=color, width=width, fill=color + (alpha,))
        if label:
            label_name = SVG_CATEGORIES[label]["name"]
            text = f"{label_name}: {score:.2f}" if score is not None else label_name
            text_bbox = draw_overlay.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_bg_x2 = x1 + text_width + 4 + 30
            text_bg_y2 = y1 - 10 + text_height + 2
            draw_overlay.rectangle([x1, y1 - 20, text_bg_x2, text_bg_y2], fill="black")
            draw_overlay.text((x1 + 2, y1 - 16), text, fill="white", font=font)
    image = Image.alpha_composite(image, overlay)
    image.save(output_path)
    print(f"图像已保存到 {output_path}")


# reverse_json_path = "data/visualization/example/reverse_json"
# pred_npy_path = "data/visualization/example"
# svg_path = "data/visualization/example/svg_gt"
# saved_path = "data/visualization/example/semantic"
# pkl_dir_path = "data/visualization/example"
# bbox_saved_dir = "data/visualization/example/instance"
# npy_file_names = os.listdir(reverse_json_path)
# file_names = [i.replace(".json","") for i in npy_file_names]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--raw_path", type=str, help="raw path with svg & reverse json of cad drawings", required=True)
    parser.add_argument(
        "--pred_path", type=str, help="path of the pred of semantic and instance from models", required=True
    )
    parser.add_argument("--save_path", type=str, help="path for saving visualization pics", required=True)
    parser.add_argument("--output_width", type=float, help="the width of output png", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    reverse_json_path = os.path.join(args.raw_path, "reverse_json")
    svg_path = os.path.join(args.raw_path, "svg_gt")
    pkl_dir_path = pred_npy_path = args.pred_path
    semantic_saved_path = os.path.join(args.save_path, "semantic")
    instance_saved_path = os.path.join(args.save_path, "instance")
    npy_file_names = os.listdir(reverse_json_path)
    file_names = [i.replace(".json", "") for i in npy_file_names]

    out_width = args.output_width
    os.makedirs(instance_saved_path, exist_ok=True)
    for file_name in tqdm(file_names, desc="Processing files"):
        file_json_path = os.path.join(reverse_json_path, file_name + ".json")
        with open(file_json_path, "r", encoding="utf-8") as file:
            svg_data = json.load(file)
        file_pred_path = os.path.join(pred_npy_path, file_name + ".npy")
        element_pred = np.load(file_pred_path)
        assert element_pred.shape[0] == len(svg_data)
        file_svg_path = os.path.join(svg_path, file_name + ".svg")
        tree = ET.parse(file_svg_path)
        root = tree.getroot()
        ns = root.tag[:-3]
        viewbox_new = root.attrib["viewBox"]
        width_svg = float(viewbox_new.split(" ")[2])
        height_svg = float(viewbox_new.split(" ")[3])
        stroke_width = "{:.2f}".format(min(width_svg, height_svg) / 1000)
        root_pred = etree.Element(
            "{http://www.w3.org/2000/svg}svg",
            nsmap={None: "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"},
        )
        root_pred.set("viewBox", viewbox_new)
        pkl_path = os.path.join(pkl_dir_path, file_name + ".pkl")
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        insts_mask = pkl_data["insts"]
        for index, (key, value) in enumerate(svg_data.items()):
            r, g, b = get_color_for_label(element_pred[index])
            r = 200
            g = 200
            b = 200
            name = get_name_for_label(element_pred[index])
            semanticId = str(get_id_for_label(element_pred[index]))
            instanceId = str(list(np.where(insts_mask[:, index] == True)[0]))
            if value["type"] == "Line":
                start_x, start_y = value["start_point"][0], value["start_point"][1]
                end_x, end_y = value["end_point"][0], value["end_point"][1]
                path = etree.SubElement(root_pred, "{http://www.w3.org/2000/svg}path")
                path.set(
                    "d",
                    f"M {start_x} {start_y} L {end_x} {end_y}",
                )
                path.set("fill", "none")
                path.set("stroke", f"rgb({r},{g},{b})")
                path.set("stroke-width", stroke_width)
                path.set("label", name)
                path.set("semanticId", semanticId)
                path.set("instanceId", instanceId)
            elif value["type"] == "Arc":
                start_x, start_y = value["start_point"][0], value["start_point"][1]
                end_x, end_y = value["end_point"][0], value["end_point"][1]
                rotation = value["rotation"]
                large_arc_flag = value["large_arc"]
                sweep = value["sweep"]
                rx = value["rx"]
                ry = value["ry"]
                path = etree.SubElement(root_pred, "{http://www.w3.org/2000/svg}path")
                path.set(
                    "d",
                    f"M {start_x},{start_y} A {rx},{ry} {rotation} {int(large_arc_flag)},{int(sweep)} {end_x},{end_y}",
                )
                path.set("fill", "none")
                path.set("stroke", f"rgb({r},{g},{b})")
                path.set("stroke-width", stroke_width)
                path.set("label", name)
                path.set("semanticId", semanticId)
                path.set("instanceId", instanceId)
            # {"type": "Circle", "cx": "106.00001", "cy": "21.50796200000002", "r": "0.75"}
            elif value["type"] == "Circle":
                cx, cy = value["cx"], value["cy"]
                r = value["r"]
                cirlce = etree.SubElement(root_pred, "{http://www.w3.org/2000/svg}circle")
                cirlce.set("cx", cx)
                cirlce.set("cy", cy)
                cirlce.set("r", r)
                cirlce.set("fill", "none")
                cirlce.set("stroke", f"rgb({r},{g},{b})")
                cirlce.set("stroke-width", stroke_width)
                cirlce.set("label", name)
                cirlce.set("semanticId", semanticId)
                cirlce.set("instanceId", instanceId)
            elif (
                value["type"] == "Ellispe"
            ):  # {"type": "Ellipse", "cx": "61.74199999999996", "cy": "65.80699999999999", "rx": "2.9", "ry": "2.2", "transform": "rotate(-0,732.492,175.307)"}
                cx, cy = value["cx"], value["cy"]
                rx, ry = value["rx"], value["ry"]
                tsfm = value["transform"]
                ellipse = etree.SubElement(root_pred, "{http://www.w3.org/2000/svg}ellipse")
                ellipse.set("cx", cx)
                ellipse.set("cy", cy)
                ellipse.set("rx", rx)
                ellipse.set("ry", ry)
                ellipse.set("fill", "none")
                ellipse.set("stroke", f"rgb({r},{g},{b})")
                ellipse.set("stroke-width", stroke_width)
                ellipse.set("label", name)
                ellipse.set("transform", tsfm)
                ellipse.set("semanticId", semanticId)
                ellipse.set("instanceId", instanceId)
        svg_visual_path = os.path.join(semantic_saved_path, "svg_visual")
        png_visual_path = os.path.join(semantic_saved_path, "png_visual")
        os.makedirs(svg_visual_path, exist_ok=True)
        os.makedirs(png_visual_path, exist_ok=True)

        output_svg_path = os.path.join(svg_visual_path, file_name + ".svg")
        with open(output_svg_path, "wb") as f:
            f.write(etree.tostring(root_pred, pretty_print=True, xml_declaration=True))
        output_png_path = os.path.join(png_visual_path, file_name + ".png")
        svg2png(url=output_svg_path, write_to=output_png_path, output_width=out_width, background_color="#ffffff")

        bbox_out_path = os.path.join(instance_saved_path, file_name + ".png")

        bboxes = pkl_data["bboxes"]
        labels = pkl_data["labels"]
        scores = pkl_data["scores"]
        scores = [i if i <= 1 else 1 for i in scores]
        colors = [tuple(SVG_CATEGORIES[i]["color"]) for i in labels]
        draw_bboxes(
            image_path=output_png_path,
            bboxes=bboxes,
            colors=colors,
            labels=labels,
            output_path=bbox_out_path,
            scores=scores,
            scale=out_width / width_svg,
        )


if __name__ == "__main__":
    main()
