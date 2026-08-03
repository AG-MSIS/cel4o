"""
Coded by ChatGPT 5.6 and Udo Frese under Udo Frese's supervision

Prototype visualizer for the column-wise conditional-expectation algorithm.

Controls
--------
Left / Right    previous / next history entry
Space           play / pause
Home / End      first / last entry
Up / Down       increase / decrease playback speed
S               save current frame as PNG
Esc             quit

Expected history format
-----------------------
history: list[dict], each entry containing

    step: int
    step_type: "init" | "q-element" | "householder"
    i, j: int | None
    L: np.ndarray, shape (n, m)
    Q: np.ndarray, shape (m, m) for cel4o-Q
    econd: float | None

Optional fields understood by the visualizer
--------------------------------------------
    row_weights or w:
        weighting vector shown as the rightmost panel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Literal
import math
import sys

import numpy as np
import pygame
from compact_matrix_sqrt.algorithms import cel4o_Q_with_history, cel4o_D_with_history, dct_matrix, adversarial


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_SIZE_Q = (1360, 540)
WINDOW_SIZE_D = (1360, 480)
BACKGROUND = (245, 255, 255)
PANEL_BACKGROUND = (255, 255, 250)
GRID_COLOR = (180, 180, 175)
TEXT_COLOR = (25, 25, 25)
MUTED_TEXT = (90, 90, 90)

POSITIVE_COLOR = (25, 25, 25)
NEGATIVE_COLOR = (205, 45, 45)
ZERO_COLOR = (225, 225, 220)

ACTIVE_COLOR = (45, 210, 120)
I_COLOR = (245, 135, 20)
J_COLOR = (125, 40, 210)
R_COLOR = (0, 150, 90)

X_COLOR = (130, 110, 40)
Y_COLOR = (115, 160, 225)
Z_COLOR = (210, 140, 210)

RANDOM_ICON_COLOR = (80, 80, 80)

MARGIN = 12
TITLE_HEIGHT = 100
ROW_LABEL_GAP = 8
W_LQ_GAP = 14
PANEL_GAP = 60
MATRIX_LABEL_HEIGHT = 34
MIN_CELL_SIZE = 12
MAX_CELL_SIZE = 55
ROW_LABEL_WIDTH = 58
LABEL_OFFSET = 32

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VisualState:
    step: int
    step_type: str
    i: int | None
    j: int | None
    L: np.ndarray
    Q: np.ndarray
    LQ: np.ndarray
    w: np.ndarray
    econd: float | None
    l4loss: float | None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatrixLayout:
    x: int
    y: int
    rows: int
    cols: int
    cell: int
    label: str

    @property
    def width(self) -> int:
        return self.cols * self.cell

    @property
    def height(self) -> int:
        return self.rows * self.cell

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            self.x + col * self.cell,
            self.y + row * self.cell,
            self.cell,
            self.cell,
        )


def compute_layouts(state: VisualState, width: int, height: int) -> list[MatrixLayout]:
    usable_height = height - TITLE_HEIGHT - 2 * MARGIN - MATRIX_LABEL_HEIGHT - 18

    shapes = [
        (state.L.shape[0], state.L.shape[1], "L (L4={:.3g})"),
        (state.Q.shape[0], state.Q.shape[1], "Q"),
        (state.LQ.shape[0], state.LQ.shape[1], "LQ (E(L4)={:.3g})"),
        (state.w.size, 1, "w"),
    ]
    usable_width = width - 2 * MARGIN - len(shapes) * PANEL_GAP

    total_cols = sum(cols for _, cols, _ in shapes)
    max_rows = max(rows for rows, _, _ in shapes)

    cell_by_width = usable_width // max(total_cols, 1)
    cell_by_height = usable_height // max(max_rows, 1)
    cell = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, cell_by_width, cell_by_height))

    total_matrix_width = sum(cols * cell for _, cols, _ in shapes) + 3 * PANEL_GAP
    start_x = max(MARGIN, (width - total_matrix_width) // 2)
    y = TITLE_HEIGHT + MARGIN + MATRIX_LABEL_HEIGHT

    result: list[MatrixLayout] = []
    x = start_x
    for rows, cols, label in shapes:
        result.append(MatrixLayout(x=x, y=y, rows=rows, cols=cols, cell=cell, label=label))
        x += cols * cell + PANEL_GAP

    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round((1.0 - t) * a + t * b) for a, b in zip(c1, c2))


def value_color(value: float, scale: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (255, 0, 255)
    if scale <= 0:
        return ZERO_COLOR

    strength = min(1.0, abs(value) / scale)
    strength = math.sqrt(strength)

    target = POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR
    return mix(ZERO_COLOR, target, strength)


def text_contrast(background: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
    return (250, 250, 248) if luminance < 120 else TEXT_COLOR


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str,
    pos: tuple[int, int], color: tuple[int, int, int] = TEXT_COLOR, anchor: str = "topleft",
) -> pygame.Rect:
    image = font.render(text, True, color)
    rect = image.get_rect()
    setattr(rect, anchor, pos)
    surface.blit(image, rect)
    return rect


def draw_random_cube(surface: pygame.Surface, rect: pygame.Rect, *,
    outline_color: tuple[int, int, int] = RANDOM_ICON_COLOR,
) -> None:
    """
    Draw a small perspective cube centered in rect.

    Three faces with different dot patterns are visible, so the symbol suggests
    randomness rather than one particular dice result.
    """
    size = max(10, round(min(rect.width, rect.height) * 0.57))

    cx, cy = rect.center

    # Perspective depth of the upper/right faces.
    depth_x = max(3, round(size * 0.28))
    depth_y = max(3, round(size * 0.22))

    front_w = size
    front_h = size

    left = cx - (front_w + depth_x) // 2
    top = cy - (front_h + depth_y) // 2 + depth_y

    # Front face corners.
    front_tl = (left, top)
    front_tr = (left + front_w, top)
    front_br = (left + front_w, top + front_h)
    front_bl = (left, top + front_h)

    # Corresponding rear/top corners.
    back_tl = (front_tl[0] + depth_x, front_tl[1] - depth_y)
    back_tr = (front_tr[0] + depth_x, front_tr[1] - depth_y)

    front_face = [front_tl, front_tr, front_br, front_bl]
    top_face = [back_tl, back_tr, front_tr, front_tl]
    right_face = [front_tr, back_tr, (front_br[0] + depth_x, front_br[1] - depth_y), front_br]

    # Slightly different shades make the 3D shape legible.
    front_color = (248, 248, 244)
    top_color = (230, 230, 226)
    right_color = (210, 210, 207)

    pygame.draw.polygon(surface, top_color, top_face)
    pygame.draw.polygon(surface, right_color, right_face)
    pygame.draw.polygon(surface, front_color, front_face)

    line_width = max(1, round(size * 0.07))
    pygame.draw.polygon(surface, outline_color, top_face, width=line_width)
    pygame.draw.polygon(surface, outline_color, right_face, width=line_width)
    pygame.draw.polygon(surface, outline_color, front_face, width=line_width)

    dot_radius = max(1, round(size * 0.075))

    def point_on_quad(quad: list[tuple[int, int]],  u: float, v: float,
    ) -> tuple[int, int]:
        """
        Bilinear interpolation inside a four-corner polygon.

        Corner order:
            top-left, top-right, bottom-right, bottom-left
        """
        tl, tr, br, bl = quad

        x = (
            (1 - u) * (1 - v) * tl[0]
            + u * (1 - v) * tr[0]
            + u * v * br[0]
            + (1 - u) * v * bl[0]
        )
        y = (
            (1 - u) * (1 - v) * tl[1]
            + u * (1 - v) * tr[1]
            + u * v * br[1]
            + (1 - u) * v * bl[1]
        )
        return round(x), round(y)

    def draw_dots(face: list[tuple[int, int]], positions: tuple[tuple[float, float], ...],
        radius: int = dot_radius,
    ) -> None:
        for u, v in positions:
            pygame.draw.circle(surface, outline_color, point_on_quad(face, u, v), radius)

    # Different patterns on the three visible faces.
    # Front: four dots
    draw_dots(front_face, ((0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)))

    # Top: three dots
    draw_dots(
        top_face, ((0.27, 0.68), (0.50, 0.50), (0.73, 0.32)), max(1, dot_radius - 1))

    # Right: two dots
    draw_dots(right_face, ((0.50, 0.30), (0.50, 0.70)), max(1, dot_radius - 1))

def draw_index_marker(surface: pygame.Surface, layout: MatrixLayout, index: int | None, *,
    axis: str, color: tuple[int, int, int], label: str, font: pygame.font.Font, gap=18
) -> None:
    if index is None:
        return
    radius = max(12, layout.cell // 3)
    if axis == "row":
        if not 0 <= index < layout.rows: return
        center = (
            layout.x - radius - gap, 
            layout.y + index * layout.cell + layout.cell // 2
        )
    elif axis == "col":
        if not 0 <= index < layout.cols: return
        center = (
            layout.x + index * layout.cell + layout.cell // 2,
            layout.y - radius - gap,
        )
    else: raise ValueError(f"Unknown marker axis: {axis!r}")

    # Farbiger Kreis
    pygame.draw.circle(surface, color, center, radius)

    # Dezente dunkle Kontur
    pygame.draw.circle(surface, mix(color, TEXT_COLOR, 0.28), center, radius, width=2)

    # Weißer, fetter Buchstabe
    draw_text(surface, font, label, center, (255, 255, 255), "center")

def draw_row_labels(surface: pygame.Surface, layout: MatrixLayout, font: pygame.font.Font,
    label:str, color: tuple[int, int, int], *, align:str="left", gap: int = 8) -> None:
    if align=="left": talign = "midright"
    elif align=="right": talign = "midleft"
    else: raise ValueError ("align must be 'left' or 'right'")
    for r in range(layout.rows):
        if align=="left": cx = layout.x - gap
        else: cx = layout.x + layout.width + gap
        cy = layout.y + r * layout.cell + layout.cell // 2
        draw_text(surface, font, label.format(r), (cx, cy), color, talign)

def draw_col_labels(surface: pygame.Surface, layout: MatrixLayout, font: pygame.font.Font,
    label:str, color: tuple[int, int, int], *,  align:str="top", gap: int = 8) -> None:
    if align=="top": talign = "midbottom"
    elif align=="bottom": talign = "midtop"
    else: raise ValueError ("align must be 'top' or 'bottom'")
    for k in range(layout.cols):
        if align=="top": cy = layout.y - gap
        else: cy = layout.y + layout.height + gap
        cx = layout.x + k * layout.cell + layout.cell // 2
        draw_text(surface, font, label.format(k), (cx, cy), color, talign)

def top_right_rect (rect: pygame.Rect, fraction: float) -> pygame.Rect:
    """
    Returns a square in the upper-right corner whose side length is
    `fraction * min(rect.width, rect.height)`.

    fraction should typically be in [0, 1].
    """
    side = round(fraction * min(rect.width, rect.height))
    return pygame.Rect(rect.right - side, rect.top, side, side)

def draw_matrix(surface: pygame.Surface,  matrix: np.ndarray, layout: MatrixLayout, *,
    label_font: pygame.font.Font, value_font: pygame.font.Font, 
    random_mask: np.ndarray | None = None, show_values: bool = True, label_val: Any = None,
    show_zero : bool = True
) -> None:
    """Draws matrix in layout with values as two digit numbers (value_font)
    and black for plus /red for minus. The label is shown below in black label_font.
    If a random_mask is passed, random symbols are shown with the entries:
       0: show number ("fixed")
       1: show number and dice icon ("partially random")
       2: only show dice icon ("fully random")
    If label_val is given, label is used as a format to render label_val.
    """
    label_pos = (layout.x + layout.width // 2, layout.y + layout.height + LABEL_OFFSET)
    mylabel = layout.label
    if label_val is not None:
        mylabel = mylabel.format (label_val)
    draw_text(surface, label_font, mylabel, label_pos, TEXT_COLOR, "midtop")

    finite = np.abs(matrix[np.isfinite(matrix)])
    scale = float(np.percentile(finite, 90)) if finite.size else 1.0
    if scale <= 1e-15:
        scale = 1.0

    for row in range(layout.rows):
        for col in range(layout.cols):
            rect = layout.cell_rect(row, col)
            rm = 0
            if random_mask is not None: rm = random_mask [row, col]
            if rm==2: color = ZERO_COLOR
            else: color = value_color(float(matrix[row, col]), scale)

            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRID_COLOR, rect, width=1)

            if rm==2: # fully random
                draw_random_cube (surface, rect)
            else:
                if show_values and layout.cell >= 28:
                    value = float(matrix[row, col])
                    if abs(value)< 0.005:
                        text = "0" if show_zero else ""
                    elif abs(value) < .995: # +.xy
                        text = f"{value:+.2f}"
                        text = text.replace("+0.", "+.").replace("-0.", "-.")
                    elif abs(value) < 10:# +x.y
                        text = f"{value:+.1f}"
                    else: # +x
                        text = f"{value:+.0f}"
                else: text = ""
                if rm==0: # fixed
                    draw_text(surface, value_font, text, rect.center, text_contrast(color), "center")
                else: # rm==1 partially random
                    draw_random_cube (surface, top_right_rect (rect, 0.65))
                    draw_text(surface, value_font, text, rect.bottomleft, text_contrast(color), "bottomleft")

def draw_submatrix_marking (surface: pygame.Surface, layout: MatrixLayout,
    rows: slice, cols:slice, *,
    color: tuple[int, int, int] = ACTIVE_COLOR, width: int | None = None,
    text_row: str | None = None, text_col: str | None = None, font: pygame.font.Font | None = None, offset: int = 0,
) -> None:
    """
    Marks submatrix in the matrix layout by a rect of color with width.
    Submatrix is a pair of index slices
    """
    r0, r1, _ = rows.indices(layout.rows)
    c0, c1, _ = cols.indices(layout.cols)
    if width is None:
        width = max(3, layout.cell // 10)
    padding = 0
    rect = pygame.Rect(
        layout.x + c0 * layout.cell - padding,
        layout.y + r0 * layout.cell - padding,
        (c1 - c0) * layout.cell + 2 * padding,
        (r1 - r0) * layout.cell + 2 * padding,
    )
    pygame.draw.rect(surface, color, rect, width=width)

    if text_col:
        if not font: raise ValueError ("font must be passed to render text")
        x = layout.x + (c0 + c1) * layout.cell / 2
        y = layout.y + layout.height + offset
        draw_text(surface, font, text_col, (round(x), round(y)), color, "midtop")
    if text_row:
        if not font: raise ValueError ("font must be passed to render text")
        x = layout.x - offset
        y = layout.y + (r0 + r1) * layout.cell / 2
        draw_text(surface, font, text_row, (round(x), round(y)), color, "midright")

def draw_diagonal_marking(surface: pygame.Surface, layout: MatrixLayout, diagonal_slice: slice, color: tuple[int, int, int], *,
                          label: str, font : pygame.font.Font, width: int = 4, padding: int = 3, offset: int = 8) -> None:
    """
    Mark a contiguous section of the main diagonal.

    The first and last diagonal cells have square, cell-aligned ends,
    while the two long sides are straight diagonal lines.
    """
    band = 0.8 * layout.cell
    half = band / 2 
    n = min(layout.rows, layout.cols)
    start, stop, step = diagonal_slice.indices(n)

    if step != 1: raise ValueError("draw_diagonal_marking only supports slices with step=1")
    first = layout.cell_rect(start, start)
    last  = layout.cell_rect(stop - 1, stop - 1)

    if start >= stop:
        # No element at all but place label
        label_pos = (layout.x + (start+stop)*layout.cell/2, 
                     layout.y + min(start+stop+1, 2*n)*layout.cell/2 + offset)
    elif start == stop - 1:
        # Only one diagonal element: ordinary rectangle.
        rect = first.inflate(2 * padding, 2 * padding)
        pygame.draw.rect(surface, color, rect, width=width)
        label_pos = (rect.centerx, rect.bottom + 4)
    else:
        # Smooth diagonal band with square-ish ends:
        #
        #       ┌────┐
        #       │    ╲
        #       │     ╲
        #       ╲      │
        #        ╲     │
        #         └────┘
        x0, y0 = first.center
        x1, y1 = last.center
        d = half / np.sqrt(2)
        points = [
            (first.left, first.top), # Ecke
            (first.left + 2*d, first.top), # Anfang Diagonale: auf oberem Rand der ersten Zelle
            (last.right, last.bottom - 2*d), # obere/rechte Begrenzung bis zum rechten Rand der letzten Zelle
            (last.right, last.bottom), # Ecke
            (last.right - 2*d, last.bottom), # Ende: auf unterem Rand der letzten Zelle
            (first.left, first.top + 2*d), # untere/linke Begrenzung diagonal zurück zum linken Rand der ersten Zelle
        ]
        pygame.draw.polygon(surface, color, points, width=width)

        # Put label below the diagonal section, centered horizontally.
        label_pos = (round((first.centerx + last.centerx) / 2),
            round((first.bottom + last.bottom)/2) + padding + 12)

    if label:draw_text(surface, font, label, label_pos, color, "midtop")

def draw_vector(surface: pygame.Surface, vector: np.ndarray, layout: MatrixLayout, *,
    label_font: pygame.font.Font, value_font: pygame.font.Font,
) -> None:
    draw_matrix(surface, vector.reshape(-1, 1), layout,
        label_font=label_font, value_font=value_font,
    )

def draw_relation_symbol (surface: pygame.Surface, left: MatrixLayout, right: MatrixLayout,
    symbol: str, font: pygame.font.Font, color: tuple[int, int, int], 
    position : literal["above"] | Literal["center"] | Literal["below"] = "center"
) -> None:
    """prints symbol between the left and right matrix layout in the given font and color.
    position defines vertically where."""
    x = round((left.x + left.width + right.x) / 2)
    if position=='above': 
        y = left.y
        text_align="midbottom"
    elif position=='center': 
        y = left.y + left.height // 2
        text_align="center"
    elif position=='below':
        y = left.y + left.height + LABEL_OFFSET
        text_align="midtop"
    else: raise ValueError("position must be above, center or below")
    draw_text(surface, font, symbol, (x, y), color, text_align)

def step_type_color(step_type: str) -> tuple[int, int, int]:
    if step_type == "q-element":
        return Y_COLOR
    if step_type == "householder":
        return X_COLOR
    return MUTED_TEXT

def random_Q_mask (n:int, i:int | None, j:int | None)->np.ndarray:
    """Returns a mask, which entries of Q are random when the
    algorithms main indices are at i, j. They are in column-major
    order all element after (i,j)"""
    mask = np.zeros (shape=(n,n), dtype=int)
    if i is None and j is None: mask[:,:] = 2 # all random
    if i is None: i=n
    if j is None: j=n
    if j<n: mask[i+1:,j] = 2
    if j+1<n: mask[j:,j+1:] = 2
    return mask

def random_LQ_mask (shape:tuple[int, int], i:int | None, j:int | None)->np.ndarray:
    """Returns a mask, which entries of LQ are (partially) random 
    when the algorithms main indices are at i, j. Colum j is
    partially random (1) and all after that are fully random (2)."""
    _, m = shape
    mask = np.zeros(shape=shape, dtype=int)
    if i is None and j is None: mask[:,:] = 2 # all random
    if i is None: i=m
    if j is None: j=m
    if j<m:
        if i<m-1: mask [:,j] = 1 # column j is partially fixed
        else: mask [:, j] = 0 # column j is fully fixed
    if j+1<m: mask [:,j+1:] = 2 # columns after j are fully random
    return mask


# ---------------------------------------------------------------------------
#  Base class for both visualizers
# ---------------------------------------------------------------------------
class HistoryVisualizer:
    def __init__(
        self, history: Sequence[dict[str, Any]], title:str = "History Visualizer", filebase:str="hist", 
        size: tuple[int, int] = None, *, fps: int = 60, seconds_per_step: float = 0.9,
    ) -> None:
        if not history:
            raise ValueError("history must not be empty")

        self.history = history
        self.states = [self.derive_visual_state(entry) for entry in history]
        self.index = 0
        self.playing = False
        self.fps = fps
        self.seconds_per_step = seconds_per_step
        self.elapsed = 0.0
        self.title = title
        self.filebase = filebase

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont("segoeui", 36, bold=True)
        self.label_font = pygame.font.SysFont("segoeui", 24, bold=True)
        self.small_font = pygame.font.SysFont("segoeui", 16)
        self.marker_font = pygame.font.SysFont("segoeui", 22, bold=True)
        self.value_font = pygame.font.SysFont("consolas", 20)
        self.econd_font = pygame.font.SysFont("segoeui", 28, bold=True)
        self.row_label_font = pygame.font.SysFont("segoeui", 18, bold=True)

    def derive_visual_state (self, history_entry:dict)->Any:
        "Overload to convert the log into the visualizers own representation"
        return history_entry

    def next_step(self) -> None:
        if self.index < len(self.states) - 1:
            self.index += 1
        else:
            self.playing = False

    def previous_step(self) -> None:
        self.index = max(0, self.index - 1)

    def save_frame(self) -> Path:
        frame_dir = Path("visualization/frames")
        frame_dir.mkdir(exist_ok=True)
        path = frame_dir / f"{self.filebase}_{self.index:05d}.png"
        pygame.image.save(self.screen, path)
        print(f"Saved {path}")
        return path

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT:
                self.playing = False
                self.next_step()
            elif event.key == pygame.K_LEFT:
                self.playing = False
                self.previous_step()
            elif event.key == pygame.K_HOME:
                self.playing = False
                self.index = 0
            elif event.key == pygame.K_END:
                self.playing = False
                self.index = len(self.states) - 1
            elif event.key == pygame.K_UP:
                self.seconds_per_step = max(0.08, self.seconds_per_step / 1.25)
            elif event.key == pygame.K_DOWN:
                self.seconds_per_step = min(8.0, self.seconds_per_step * 1.25)
            elif event.key == pygame.K_s:
                self.save_frame()
        return True

    def update(self, dt: float) -> None:
        if not self.playing:
            self.elapsed = 0.0
            return

        self.elapsed += dt
        while self.elapsed >= self.seconds_per_step:
            self.elapsed -= self.seconds_per_step
            self.next_step()

    def draw(self) -> None:
        surface = self.screen
        width, height = surface.get_size()
        surface.fill(BACKGROUND)


# ---------------------------------------------------------------------------
# Visualizer for cel4o-Q
# ---------------------------------------------------------------------------

class QHistoryVisualizer (HistoryVisualizer):
    def __init__(
        self, history: Sequence[dict[str, Any]], *,
        filebase: str = "cel4o-q",
        size: tuple[int, int] = WINDOW_SIZE_Q, fps: int = 60, seconds_per_step: float = 0.9,
    ) -> None:
        super().__init__(history, "Conditional-expectation L4 optimization of Q", filebase,
            size=size, fps=fps, seconds_per_step=seconds_per_step)

    def derive_visual_state (self, entry: dict[str, Any]) -> VisualState:
        """
        Convert one history entry into everything needed by the renderer.

        The automatically derived masks are deliberately simple. This is the main
        function to adapt once the exact x/y/z and random-region definitions are
        fixed.
        """
        L = np.asarray(entry["L"], dtype=float)
        Q = np.asarray(entry["Q"], dtype=float)

        if L.ndim != 2 or Q.ndim != 2:
            raise ValueError("L and Q must be matrices")
        m = L.shape[1]
        if Q.shape != (m, m):
            raise ValueError(f"Incompatible shapes L={L.shape}, Q={Q.shape}")

        LQ = L @ Q
        i = entry.get("i")
        j = entry.get("j")

        w_value = entry.get("row_weights", entry.get("w"))
        if w_value is None:
            # A neutral placeholder. Replace this with the actual definition of w.
            w = np.ones(L.shape[0], dtype=float)
        else:
            w = np.asarray(w_value, dtype=float).reshape(-1)
        if w.size != L.shape[0]:
            raise ValueError("row weights must have one entry per row of L")

        return VisualState(step=int(entry.get("step", 0)), step_type=str(entry.get("step_type", "unknown")),
            i=int(i) if isinstance(i, (int, np.integer)) else None,
            j=int(j) if isinstance(j, (int, np.integer)) else None,
            L=L, Q=Q, LQ=LQ, w=w,
            econd=None if entry.get("econd") is None else float(entry["econd"]),
            l4loss=None if entry.get("l4loss") is None else float(entry["l4loss"])
        )

    def draw(self) -> None:
        surface = self.screen
        width, height = surface.get_size()
        state = self.states[self.index]
        m = state.L.shape[1]

        surface.fill(BACKGROUND)

        layouts = compute_layouts(state, width, height)
        l_layout, q_layout, lq_layout, w_layout = layouts

        # Upper part
        draw_text(surface, self.title_font, self.title, (5, 24), TEXT_COLOR, "topleft")
        type_color = step_type_color(state.step_type)
        if state.step_type=="final": text = "final"
        else: text = "after "+state.step_type+" step"
        draw_text(surface, self.econd_font, text, 
                  (width-5, 24), type_color, anchor="topright")

        # Matrices with various markings
        draw_matrix(surface, state.L, l_layout, label_font=self.label_font, value_font=self.value_font,
                    label_val=state.l4loss)
        draw_matrix(surface, state.Q, q_layout, label_font=self.label_font, value_font=self.value_font,
            random_mask = random_Q_mask(m, state.i, state.j)
        )
        draw_relation_symbol (surface, q_layout, lq_layout, "=", self.title_font, TEXT_COLOR, "center")
        draw_relation_symbol (surface, q_layout, lq_layout, "=", self.label_font, TEXT_COLOR, "below")
        draw_relation_symbol (surface, l_layout, q_layout, "·", self.title_font, TEXT_COLOR, "center")
        draw_relation_symbol (surface, l_layout, q_layout, "·", self.label_font, TEXT_COLOR, "below")
        draw_matrix(surface, state.LQ, lq_layout, label_font=self.label_font, value_font=self.value_font,
                    random_mask=random_LQ_mask(state.LQ.shape, state.i, state.j), label_val=state.econd
        )
        draw_vector(surface, state.w, w_layout, label_font=self.label_font, value_font=self.value_font)
        draw_submatrix_marking (surface, q_layout, slice(state.j, None), slice(state.j, None ), 
                                color=ACTIVE_COLOR, text_col="active", font=self.label_font)
        if isinstance (state.i, int) and isinstance(state.j, int):
            draw_submatrix_marking (surface, l_layout, slice(None, None), slice(state.j, state.i), 
                                    color=X_COLOR, text_col="a", font=self.label_font)
            draw_submatrix_marking (surface, l_layout, slice(None, None), slice(state.i, state.i+1), 
                                    color=Y_COLOR, text_col="b", font=self.label_font)
            draw_submatrix_marking (surface, l_layout, slice(None, None), slice(state.i+1, None), 
                                    color=Z_COLOR, text_col="c", font=self.label_font, offset=0 if state.i<m else 15)
            draw_submatrix_marking (surface, q_layout, slice(state.j, state.i), slice(state.j, state.j+1),
                                    color=X_COLOR, text_row = "x", font=self.label_font)
            draw_submatrix_marking (surface, q_layout, slice(state.i, state.i+1), slice(state.j, state.j+1),
                                    color=Y_COLOR, text_row = "y", font=self.label_font, offset=0 if state.i<m else 12)
            draw_submatrix_marking (surface, q_layout, slice(state.i+1, None), slice(state.j, state.j+1),
                                    color=Z_COLOR, text_row = "z", font=self.label_font)
            draw_index_marker(surface, q_layout, state.i, axis="row",
                color=I_COLOR, label="i", font=self.marker_font
            )
            draw_index_marker(surface, l_layout, state.i, axis="col",
                color=I_COLOR, label="i", font=self.marker_font
            )
            draw_index_marker(surface, q_layout, state.j, axis="col",
                color=J_COLOR, label="j", font=self.marker_font
            )
            draw_index_marker(surface, lq_layout, state.j, axis="col",
                color=J_COLOR, label="j", font=self.marker_font
            )
        draw_row_labels (surface, w_layout, self.row_label_font, "r={}", R_COLOR, gap=12)
        draw_row_labels (surface, l_layout, self.row_label_font, "r={}", R_COLOR, gap=4)

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.fps) / 1000.0
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self.update(dt)
            self.draw()

        pygame.quit()

def main_Q_square() -> None:
    n = m = 6
    L = np.eye(n, m)
    _, _, history = cel4o_Q_with_history (L)
    QHistoryVisualizer(history, filebase="cel4o-q-square").run()

def main_Q_rect() -> None:
    n, m = 4, 6
    L = np.eye(n, m)
    _, _, history = cel4o_Q_with_history (L)
    QHistoryVisualizer(history, filebase="cel4o-q-rect").run()

## Visualization of conditonal expectation l4-optimization of D
@dataclass
class DVisualState:
    step: int
    step_type: str
    k: int | None
    L: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    LDQ: np.ndarray
    W: np.ndarray
    signs: np.ndarray
    econd: float | None
    l4loss: float | None

    @property
    def has_random_remainder(self) -> bool:
        return bool(np.any(self.signs == 0))

def compute_signs_layouts(state: DVisualState, width: int, height: int, *,
    margin: int = 24, title_height: int = 110, matrix_label_height: int = 48,
    factor_gap: int = 52, equality_gap: int = 60, min_cell: int = 12, max_cell: int = 55,
) -> tuple[MatrixLayout, MatrixLayout, MatrixLayout, MatrixLayout, MatrixLayout]:
    shapes = [
        (state.L.shape[0], state.L.shape[1], "L"),
        (state.D.shape[0], state.D.shape[1], "D"),
        (state.Q.shape[0], state.Q.shape[1], "Q (L4(LQ)={:.3g})"),
        (state.LDQ.shape[0], state.LDQ.shape[1], "LDQ (E(L4)={:.3g})"),
        (state.W.size, 1, "W"),
    ]
    gaps = [factor_gap, factor_gap, equality_gap, factor_gap]

    usable_width = width - 2 * margin - sum(gaps)
    usable_height = height - title_height - 2 * margin - matrix_label_height
    total_cols = sum(cols for _, cols, _ in shapes)
    max_rows = max(rows for rows, _, _ in shapes)
    cell = max(min_cell, min(max_cell, usable_width // total_cols, usable_height // max_rows))

    total_width = sum(cols * cell for _, cols, _ in shapes) + sum(gaps)
    margin_left = margin+16
    x = max(margin_left, (width - total_width) // 2)
    y = title_height + margin

    layouts: list[MatrixLayout] = []
    for idx, (rows, cols, label) in enumerate(shapes):
        layouts.append(MatrixLayout(x=x, y=y, rows=rows, cols=cols, cell=cell, label=label))
        x += cols * cell
        if idx < len(gaps):
            x += gaps[idx]
    return tuple(layouts)  # type: ignore[return-value]

def random_D_mask (n:int, k:int | None)->np.ndarray:
    """Returns a mask, which entries of D are random when the
    algorithms main index is at k. They are the diagonal entries
    following row/column k."""
    mask = np.zeros((n,n), dtype=int)
    if k is None: k = -1
    if isinstance(k, int):
        for i in range(k+1, n): mask[i,i]=2
    return mask

def random_LDQ_mask (L:Matrix, output_shape:tuple[int, int], k:int | None)->np.ndarray:
    """Returns a mask, which entries of LDQ are random when the
    algorithms main index is at k. The algorithm considers the
    sparsity pattern of L."""
    p = L.shape[1]
    mask = np.zeros(output_shape, dtype=int)
    if k is None: 
        mask[:,:] = 2
    elif isinstance(k, int):
        for j in range(k+1, p):
            mask[L[:,j]!=0,:] = 1
    return mask

class DHistoryVisualizer (HistoryVisualizer):
    """Second visualizer; shares all low-level drawing functions with cel4o-Q."""

    def __init__(self, history: Sequence[dict[str, Any]], *,
        filebase: str = "cel4o-d",
        size: tuple[int, int] = WINDOW_SIZE_D, fps: int = 60, seconds_per_step: float = 0.9,
    ) -> None:
        super().__init__(history,
            "Conditional-expectation L4 optimization of diagonal signs D", filebase,
            size=size, fps=fps, seconds_per_step=seconds_per_step)

    def derive_visual_state (self, entry: dict[str, Any]) -> DVisualState:
        L = np.asarray(entry["L"], dtype=float)
        Q = np.asarray(entry["Q"], dtype=float)
        signs = np.asarray(entry["signs"], dtype=float).reshape(-1)
        LDQ = np.asarray(entry["LDQ"], dtype=float)

        if L.ndim != 2 or Q.ndim != 2 or LDQ.ndim != 2:
            raise ValueError("L, Q and LDQ must be matrices")
        if not np.all(np.isin(signs, (-1.0, 0.0, 1.0))):
            raise ValueError("signs must contain only -1, 0, +1")
        if L.shape[1] != signs.size or Q.shape[0] != signs.size:
            raise ValueError("L, signs and Q have incompatible dimensions")
        if LDQ.shape != (L.shape[0], Q.shape[1]):
            raise ValueError("LDQ has the wrong shape")

        w = entry.get("row_weights", entry.get("w"))
        W = np.ones(L.shape[0]) if w is None else np.asarray(w, dtype=float).reshape(-1)
        if W.size != L.shape[0]:
            raise ValueError("row weights must have one entry per row of L")

        k_value = entry.get("k")
        k = int(k_value) if isinstance(k_value, (int, np.integer)) else None

        return DVisualState(
            step=int(entry.get("step", 0)),
            step_type=str(entry.get("step_type", "unknown")),
            k=k, L=L, D=np.diag(signs), Q=Q, LDQ=LDQ, W=W, signs=signs,
            econd=None if entry.get("econd") is None else float(entry["econd"]),
            l4loss=None if entry.get("l4loss") is None else float(entry["l4loss"]),
        )

    def draw(self) -> None:
        surface = self.screen
        width, height = surface.get_size()
        state = self.states[self.index]
        surface.fill(BACKGROUND)
        p = state.L.shape[1]

        draw_text(surface, self.title_font, self.title, (MARGIN, 24), TEXT_COLOR, "topleft")

        l_layout, d_layout, q_layout, ldq_layout, w_layout = compute_signs_layouts(
            state, width, height
        )

        right = w_layout.x + w_layout.width
        draw_text(surface, self.econd_font, state.step_type, (right, 49), Y_COLOR, "topright")

        draw_matrix(surface, state.L, l_layout, label_font=self.label_font, value_font=self.value_font)
        draw_matrix(surface, state.D, d_layout, label_font=self.label_font, value_font=self.value_font,
                    random_mask=random_D_mask(p, state.k), show_zero=False)
        draw_matrix(surface, state.Q, q_layout, label_font=self.label_font, value_font=self.value_font,
                    label_val=state.l4loss)
        draw_matrix(surface, state.LDQ, ldq_layout, label_font=self.label_font, value_font=self.value_font,
                    label_val=state.econd,
                    random_mask=random_LDQ_mask(state.L, state.LDQ.shape, state.k))
        draw_matrix(surface, state.W.reshape(-1, 1), w_layout, label_font=self.label_font, value_font=self.value_font)

        draw_relation_symbol(surface, l_layout, d_layout, "·", self.title_font, TEXT_COLOR)
        draw_relation_symbol(surface, l_layout, d_layout, "·", self.label_font, TEXT_COLOR, position="below")
        draw_relation_symbol(surface, d_layout, q_layout, "·", self.title_font, TEXT_COLOR)
        draw_relation_symbol(surface, d_layout, q_layout, "·", self.label_font, TEXT_COLOR, position = "below")
        draw_relation_symbol(surface, q_layout, ldq_layout, "=", self.title_font, TEXT_COLOR)
        draw_relation_symbol(surface, q_layout, ldq_layout, "=", self.label_font, TEXT_COLOR, position="below")

        if state.k is not None:
            draw_index_marker(surface, l_layout, state.k, axis="col", color=Y_COLOR, 
                                label="k", font=self.marker_font)
            draw_index_marker(surface, d_layout, state.k, axis="row", color=Y_COLOR, label="k", 
                                font=self.marker_font, gap=6)
            draw_index_marker(surface, d_layout, state.k, axis="col", color=Y_COLOR, label="k", 
                                font=self.marker_font,)
            draw_row_labels (surface, l_layout, self.row_label_font, "i={}", I_COLOR, align="left")
            draw_col_labels (surface, q_layout, self.row_label_font, "j={}", J_COLOR, align="top")
            draw_row_labels (surface, ldq_layout, self.row_label_font, "i={}", I_COLOR, align="right")
            draw_col_labels (surface, ldq_layout, self.row_label_font, "j={}", J_COLOR, align="top")
            
            draw_index_marker(surface, q_layout, state.k, axis="row", color=Y_COLOR, label="k", 
                                font=self.marker_font, gap=16)
            draw_submatrix_marking (surface, l_layout, slice(None), slice(0,state.k), color=X_COLOR,
                                    text_col="X", font = self.label_font)
            draw_submatrix_marking (surface, l_layout, slice(None), slice(state.k,state.k+1), color=Y_COLOR,
                                    text_col="Y", font = self.label_font)
            draw_submatrix_marking (surface, l_layout, slice(None), slice(state.k+1,p), color=Z_COLOR,
                                    text_col="Z", font = self.label_font, offset=0 if state.k<p else 15)

            draw_diagonal_marking (surface, d_layout, slice(0, state.k), X_COLOR,
                                   label="X", font=self.label_font)
            draw_diagonal_marking (surface, d_layout, slice(state.k, state.k+1), Y_COLOR,
                                   label="Y", font=self.label_font)
            draw_diagonal_marking (surface, d_layout, slice(state.k+1, p), Z_COLOR,
                                   label="Z", font=self.label_font, offset=8 if state.k<p else 23)

            draw_submatrix_marking (surface, q_layout, slice(0,state.k), slice(None), color=X_COLOR,
                                    text_row="X", font = self.label_font)
            draw_submatrix_marking (surface, q_layout, slice(state.k,state.k+1), slice(None), color=Y_COLOR,
                                    text_row="Y", font = self.label_font)
            draw_submatrix_marking (surface, q_layout, slice(state.k+1,p), slice(None), color=Z_COLOR,
                                    text_row="Z", font = self.label_font, offset=0 if state.k<p else 15)

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.fps) / 1000.0
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self.update(dt)
            self.draw()
        pygame.quit()

def main_D_square():
    n = 5
    Q = dct_matrix (n, True).T
    L =  adversarial (Q)
    _, _, _, history = cel4o_D_with_history (L, Q)
    DHistoryVisualizer(history, filebase="cel4o-d-square").run()

def main_D_rect():
    n, m = 4, 7
    Q = dct_matrix ((m,n), True).T
    L =  adversarial (Q)
    _, _, _, history = cel4o_D_with_history (L, Q)
    DHistoryVisualizer(history, filebase="cel4o-d-rect").run()


if __name__ == "__main__":
    #main_Q_square()
    #main_Q_rect()
    #main_D_square()
    main_D_rect()
