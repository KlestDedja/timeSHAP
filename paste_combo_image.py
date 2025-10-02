import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


import os


def get_images_from_paths(idx, key_list, **folders):
    """
    Collects image file paths for survival curves and SHAP plots.
    ----------
    idx : int
        The index of the instance to fetch images for.
    folders : dict
        Keyword arguments specifying folder paths. Possible keys:
        - survival_curves
        - local_shap
        - local interval_shap
        - global interval_shap
    Returns: a list of image file paths that exist.
    """

    list_of_file_paths = []

    # survival curve
    if "survival_curves" in folders:
        surv_path = os.path.join(
            folders["survival_curves"], f"survival_curve_idx{idx}.png"
        )
        if os.path.exists(surv_path):
            list_of_file_paths.append(surv_path)
            print("Adding survival curve image")
        else:
            print("Survival curve image not found")

    # local SHAP
    if "local_shap" in folders:
        local_shap_path = os.path.join(
            folders["local_shap"], f"Local_SHAP_idx{idx}.png"
        )
        if os.path.exists(local_shap_path):
            list_of_file_paths.append(local_shap_path)
            print("Adding local SHAP image")
        else:
            print("Local SHAP image not found")

    # global SHAP
    if "global_shap" in folders:
        global_shap_path = os.path.join(folders["global_shap"], f"Global_SHAP.png")
        if os.path.exists(global_shap_path):
            list_of_file_paths.append(global_shap_path)
            print("Adding global SHAP image")
        else:
            print("Global SHAP image not found")

    # interval SHAP (local + global)
    if "local_interval_shap" in folders:
        # local interval
        local_shap_interval_paths = [
            os.path.join(
                folders["local_interval_shap"], f"Local_SHAP_idx{idx}_T{key}.png"
            )
            for key in key_list
        ]
        found_local = [p for p in local_shap_interval_paths if os.path.exists(p)]
        if found_local:
            list_of_file_paths.extend(found_local)
            print("Adding local SHAP interval images")
        else:
            print("No local SHAP interval images found")

    if "global_interval_shap" in folders:
        # global interval
        global_interval_paths = [
            os.path.join(
                folders["global_interval_shap"], f"Global_interval_SHAP_T{key}.png"
            )
            for key in key_list
        ]
        found_global = [p for p in global_interval_paths if os.path.exists(p)]
        if found_global:
            list_of_file_paths.extend(found_global)
            print("Adding global SHAP interval images")
        else:
            print("No global SHAP interval images found")

    return list_of_file_paths


def get_image_file_paths_old(
    idx, survival_curves_folder, local_shap_folder, interval_shap_folder
):

    list_of_file_paths = []
    # add survival curve image path (if it exists)
    try:
        surv_path = os.path.join(survival_curves_folder, f"survival_curve_idx{idx}.png")
        list_of_file_paths.append(surv_path)
        print("Adding survival curve image")
    except FileNotFoundError as e:
        print("Survival curve image not found")

    # add local shap image path (if it exists)
    try:
        local_shap_path = os.path.join(local_shap_folder, f"Local_SHAP_idx{idx}.png")
        list_of_file_paths.append(local_shap_path)
        print("Adding local SHAP image")
    except FileNotFoundError as e:
        print("Local SHAP image not found")

    # add interval shap image paths (if it exists)
    try:
        local_shap_interval_paths = [
            os.path.join(interval_shap_folder, f"Local_SHAP_idx{idx}_T{key}.png")
            for key in sorted(os.listdir(interval_shap_folder))
        ]
        list_of_file_paths.extend(local_shap_interval_paths)
        print("Adding local SHAP interval images")
    except FileNotFoundError as e:
        print("Local SHAP interval images not found")

    # add global shap image paths (if it exists)
    try:
        global_interval_paths = [
            os.path.join(interval_shap_folder, f"Global_SHAP_idx{idx}_T{key}.png")
            for key in sorted(os.listdir(interval_shap_folder))
        ]
        list_of_file_paths.extend(global_interval_paths)
        print("Adding global SHAP interval images")
    except FileNotFoundError as e:
        print("Global SHAP interval images not found")

    return list_of_file_paths


def load_images(*image_paths):
    """
    Accepts any number of image paths and returns a list of loaded images.
    Skips paths that do not exist.
    """
    return [Image.open(p) for p in image_paths if os.path.exists(p)]


def calculate_layout(
    top_images,
    bottom_images,
    max_admitted_per_row,
    x_pad_intrarow=0,
    y_pad_top=10,
    y_pad_intrarow=100,
):
    bottom_widths, bottom_heights = (
        zip(*(im.size for im in bottom_images)) if bottom_images else ([], [])
    )
    n_interval_imgs = len(bottom_widths)
    MAX_ADMITTED_PER_ROW = max_admitted_per_row

    top_widths, top_heights = (
        zip(*(im.size for im in top_images)) if top_images else ([], [])
    )

    top_required_width, top_required_height = (sum(top_widths), max(top_heights))

    if n_interval_imgs == 0:
        n_interval_rows = 0
        max_imgs_per_row = 0
    else:
        n_interval_rows = (n_interval_imgs - 1) // MAX_ADMITTED_PER_ROW + 1
        max_imgs_per_row = int(np.ceil(n_interval_imgs / n_interval_rows))
    if n_interval_imgs > 0:
        avg_width = int(np.mean(bottom_widths))
        bottom_required_width = avg_width * max_imgs_per_row + x_pad_intrarow * max(
            0, max_imgs_per_row - 1
        )
        bottom_row_heights = max(bottom_heights)
    else:
        bottom_required_width = 0
        bottom_row_heights = 0
    combo_width = max(top_required_width, bottom_required_width)
    combo_height = (
        y_pad_top
        + top_required_height
        + n_interval_rows * bottom_row_heights
        + n_interval_rows * y_pad_intrarow
    )
    return {
        "combo_width": combo_width,
        "combo_height": combo_height,
        "top_widths": top_widths,
        "top_heights": top_heights,
        "bottom_widths": bottom_widths,
        "bottom_heights": bottom_heights,
        "n_interval_imgs": n_interval_imgs,
        "n_interval_rows": n_interval_rows,
        "max_imgs_per_row": max_imgs_per_row,
        "y_pad_top": y_pad_top,
        "y_pad_intrarow": y_pad_intrarow,
        "x_pad_intrarow": x_pad_intrarow,
    }


def compute_top_row_x_pos(combo_width, img_widths, edge_padding=10):
    """
    Compute horizontal positions for placing images in the top row:
    - If only one image, center it.
    - If two images and both are bottom images, place them at the edges.
    - Otherwise, spread images evenly across the available width.
    Returns: list[int]
        List of x positions where each image should be placed.
    """
    n_imgs = len(img_widths)

    if n_imgs == 0:
        raise ValueError("No images provided for positioning.")

    if n_imgs == 1:
        # Center the single image
        w = img_widths[0]
        return [(combo_width - w) // 2]

    # General case: n_img > 1. spread images evenly
    total_img_width = sum(img_widths)
    gaps = n_imgs + 1  # gap between images, with some gap from the edges
    gap_size = max(edge_padding, (combo_width - total_img_width) // gaps)

    positions = []
    current_x = gap_size
    for w in img_widths:
        positions.append(current_x)
        current_x += w + gap_size

    return positions


def place_top_row(canvas, top_images, positions, y_pad_top):

    if len(top_images) != len(positions):
        raise ValueError(
            f"len(top_images): {len(top_images)} and len(positions): {len(positions)} do not correspond"
        )
    for image, pos in zip(top_images, positions):
        canvas.paste(image, (pos, y_pad_top))


def place_top_row_old(canvas, surv_image, local_image, positions, y_pad_top):
    canvas.paste(surv_image, (positions[0], y_pad_top))
    canvas.paste(local_image, (positions[1], y_pad_top))


def place_bottom_rows(canvas, bottom_images, layout):
    x_offset = 0
    # place bottom rows below existing top row, leave vertical space and additional padding
    y_offset = (
        max(layout["top_heights"])
        + layout["y_pad_top"]
        + (layout["y_pad_intrarow"] if layout["n_interval_rows"] > 0 else 0)
    )
    for idx, img in enumerate(bottom_images):
        row_number = (
            idx // layout["max_imgs_per_row"] if layout["max_imgs_per_row"] else 0
        )
        # once the (bottom) row number is identified (indexing from 0),
        # start placing the bottom images row by row
        if layout["max_imgs_per_row"] and idx % layout["max_imgs_per_row"] == 0:
            x_offset = 0
            row_width_sum = sum(
                layout["bottom_widths"][idx : idx + layout["max_imgs_per_row"]]
            )
            n_row_items = len(
                layout["bottom_widths"][idx : idx + layout["max_imgs_per_row"]]
            )
            extra_intrarow_gap = (
                (layout["combo_width"] - row_width_sum) // (n_row_items + 1)
                if n_row_items > 0
                else 0
            )
        x_curr_offset = x_offset + (
            extra_intrarow_gap if layout["max_imgs_per_row"] else 0
        )
        y_curr_offset = y_offset + row_number * (
            max(layout["bottom_heights"]) + layout["y_pad_intrarow"]
        )
        canvas.paste(img, (x_curr_offset, y_curr_offset))
        x_offset += img.size[0]
        x_offset += layout["x_pad_intrarow"]
        if layout["max_imgs_per_row"]:
            x_offset += extra_intrarow_gap


def render_title(combo_width, title, image_dpi_res, title_padding=20):
    font_size = round(36 * (image_dpi_res / 72))
    font = ImageFont.truetype("arial.ttf", font_size)

    tmp_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    try:
        bbox = tmp_draw.textbbox((0, 0), title, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = tmp_draw.textsize(title, font=font)  # type: ignore
    title_image = Image.new(
        "RGB", (combo_width, int(text_height + title_padding)), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(title_image)
    text_x = (title_image.width - text_width) / 2
    text_y = (title_image.height - text_height) / 2
    draw.text((text_x, text_y), title, fill="black", font=font)
    return title_image


def assemble_final_image(
    title_image, collage_image, combo_width, combo_height, title_padding
):
    final_image = Image.new(
        "RGB", (combo_width, title_image.height + combo_height), color=(255, 255, 255)
    )
    final_image.paste(collage_image, (0, title_image.height - title_padding))
    final_image.paste(title_image, (0, 0))
    return final_image


def save_final_image(final_image, out_dir, out_name, dpi):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    final_image.save(out_path, dpi=(dpi, dpi))
    return out_path


def display_image(final_image):
    try:
        from IPython.display import display as _display

        _display(final_image)
    except Exception:
        pass


def generate_collage_image(
    idx: int,
    # general_figs_folder: str,
    survival_figs_folder: str,
    local_shap_folder: str,
    interval_figs_folder: str,
    local_plt_name_png: str,
    interval_shap_values: dict,
    combo_local_plt_name_pdf: str,
    combo_local_plt_name_png: str,
    image_dpi_res: int = 90,
    title: str = "Time-SHAP explanation",
    max_admitted_per_row: int = 4,
    y_pad_top: int = 10,
    y_pad_intrarow: int = 100,
    x_pad_intrarow: int = 0,
    show: bool = False,
) -> tuple[Image.Image, str, str]:
    # step 1: collect paths where images are stored and load them

    interval_keys_list = list(interval_shap_values.keys())

    surv_path, local_path, interval_paths = get_images_from_paths(
        idx,
        interval_keys_list,
        survival_figs_folder,
        local_shap_folder,
        interval_figs_folder,
    )
    all_image_paths = [surv_path, local_path] + interval_paths

    # all_image_paths = get_image_file_paths(
    #     idx,
    #     general_figs_folder,
    #     interval_figs_folder,
    #     local_plt_name_png,
    #     interval_shap_values,
    # )
    images = load_images(*all_image_paths)

    # Step 2: assign images by position, assumes 1st image is survival plot, 2nd image
    # is local SHAP over entire interval
    # Rest of the images are local SHAP intervals
    # surv_image = images[0] if len(images) > 0 else None
    # local_image = images[1] if len(images) > 1 else None
    top_row_images = images[:2]
    interval_images = images[2:] if len(images) > 2 else []
    layout = calculate_layout(
        top_row_images,
        interval_images,
        max_admitted_per_row,
        x_pad_intrarow,
        y_pad_top,
        y_pad_intrarow,
    )
    combo_image = Image.new(
        "RGB", (layout["combo_width"], layout["combo_height"]), color=(255, 255, 255)
    )
    positions = compute_top_row_x_pos(
        layout["combo_width"],
        [layout["surv_w"], layout["local_w"]],
        layout["max_imgs_per_row"] if layout["max_imgs_per_row"] else 0,
    )
    place_top_row(
        combo_image, top_row_images[0], top_row_images[1], positions, y_pad_top
    )
    place_bottom_rows(combo_image, interval_images, layout, layout["combo_width"])
    # clean up
    # if local_image:
    #     local_image.close()
    # if surv_image:
    #     surv_image.close()
    for im in top_row_images:
        im.close()
    for im in interval_images:
        im.close()

    # add title and store everything
    title_image, v_padding = render_title(layout["combo_width"], title, image_dpi_res)
    final_image = assemble_final_image(
        title_image,
        combo_image,
        layout["combo_width"],
        layout["combo_height"],
        v_padding,
    )
    out_dir = os.path.join(general_figs_folder, "combo-plots")
    pdf_path, png_path = save_final_image(
        final_image,
        out_dir,
        combo_local_plt_name_pdf,
        combo_local_plt_name_png,
        image_dpi_res,
    )
    if show:
        display_image(final_image)
    return final_image, pdf_path, png_path


# def collage_local_image(
#     idx: int,
#     general_figs_folder: str,
#     interval_figs_folder: str,
#     local_plt_name_png: str,
#     interval_shap_values: dict,
#     combo_local_plt_name_pdf: str,
#     combo_local_plt_name_png: str,
#     *,
#     image_dpi_res: int = 90,
#     title: str = "Time-SHAP explanation",
#     max_admitted_per_row: int = 4,
#     y_pad_top: int = 10,
#     y_pad_intrarow: int = 100,
#     x_pad_intrarow: int = 0,
#     show: bool = False,
# ) -> tuple[Image.Image, str, str]:

#     # step 1: collect paths where images are stored
#     surv_path, local_path, interval_paths = get_image_paths(
#         idx,
#         general_figs_folder,
#         interval_figs_folder,
#         local_plt_name_png,
#         interval_shap_values,
#     )
#     # step 2: load images
#     surv_image, local_image, images = load_images(surv_path, local_path, interval_paths)
#     layout = calculate_layout(
#         images,
#         surv_image,
#         local_image,
#         max_admitted_per_row,
#         x_pad_intrarow,
#         y_pad_top,
#         y_pad_intrarow,
#     )
#     # create combno image after computing needed size
#     combo_image = Image.new(
#         "RGB", (layout["combo_width"], layout["combo_height"]), color=(255, 255, 255)
#     )
#     # compute positions for images within the combo image
#     positions = compute_x_positions(
#         layout["combo_width"],
#         [layout["surv_w"], layout["local_w"]],
#         layout["max_imgs_per_row"] if layout["max_imgs_per_row"] else 0,
#     )
#     # place the images of the top row (survival curve and local SHAP)
#     place_top_row(combo_image, surv_image, local_image, positions, y_pad_top)
#     # place local interval shap images along the other rows
#     place_bottom_rows(combo_image, images, layout, layout["combo_width"])
#     local_image.close()
#     surv_image.close()

#     # clean up
#     for im in images:
#         im.close()

#     # add title and store everything
#     title_image, v_padding = render_title(layout["combo_width"], title, image_dpi_res)
#     final_image = assemble_final_image(
#         title_image,
#         combo_image,
#         layout["combo_width"],
#         layout["combo_height"],
#         v_padding,
#     )
#     out_dir = os.path.join(general_figs_folder, "combo-plots")
#     pdf_path, png_path = save_final_image(
#         final_image,
#         out_dir,
#         combo_local_plt_name_pdf,
#         combo_local_plt_name_png,
#         image_dpi_res,
#     )
#     if show:
#         display_image(final_image)
#     return final_image, pdf_path, png_path
