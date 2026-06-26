import cv2

_BOX_COLOR = (0, 0, 255)
_BOX_THICKNESS = 2
_LABEL_SCALE = 0.6
_LABEL_THICKNESS = 2
_LABEL_PADDING = 4


def annotate_frame(frame, bbox: list[float], label: str | None = None):
    annotated = frame.copy()
    if not bbox or len(bbox) < 4:
        return annotated

    x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
    height, width = annotated.shape[:2]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

    if not label:
        return annotated

    text = str(label)
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_SCALE,
        _LABEL_THICKNESS,
    )
    label_top = max(0, y1 - text_height - baseline - _LABEL_PADDING * 2)
    label_bottom = label_top + text_height + baseline + _LABEL_PADDING * 2
    label_right = min(width, x1 + text_width + _LABEL_PADDING * 2)
    cv2.rectangle(annotated, (x1, label_top), (label_right, label_bottom), _BOX_COLOR, -1)
    cv2.putText(
        annotated,
        text,
        (x1 + _LABEL_PADDING, label_bottom - baseline - _LABEL_PADDING),
        cv2.FONT_HERSHEY_SIMPLEX,
        _LABEL_SCALE,
        (255, 255, 255),
        _LABEL_THICKNESS,
        cv2.LINE_AA,
    )
    return annotated
