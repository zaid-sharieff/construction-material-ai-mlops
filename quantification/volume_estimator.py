from quantification.density_constants import DENSITY

# NOTE:
# Height is estimated proportionally due to lack of depth data.
# This assumes a bounded heap or container-like placement.

PIXEL_TO_METER = 0.0005      # Approximate pixel-to-meter scale
HEIGHT_RATIO = 0.30         # Assumed height as a fraction of base width

def estimate_volume_from_bbox(bbox):
    x1, y1, x2, y2 = bbox

    width_px = abs(x2 - x1)
    depth_px = abs(y2 - y1)

    width_m = width_px * PIXEL_TO_METER
    depth_m = depth_px * PIXEL_TO_METER

    base_area = width_m * depth_m
    height_m = width_m * HEIGHT_RATIO

    volume = base_area * height_m
    return volume

def estimate_mass(volume_m3, material):
    density = DENSITY.get(material, 1500)
    return volume_m3 * density
