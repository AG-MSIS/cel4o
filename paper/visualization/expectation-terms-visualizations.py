import io
from dataclasses import dataclass

import pygame
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

@dataclass
class Term:
    alpha: int = 0
    beta: int = 0

    # Multiplicities of the different Z variables.
    #
    # (2,)     -> Z_1^2
    # (1, 1)   -> Z_1 Z_2
    # (2, 1)   -> Z_1^2 Z_2
    # (4,)     -> Z_1^4
    z_groups: tuple[int, ...] = ()

    expectation: str = ""
    delta_beta: str = ""

terms1 = [
    # only Z_kc_k terms
    Term(z_groups=(1,1,1,1)),
    Term(z_groups=(2,1,1)),
    Term(z_groups=(2,2), expectation=r'$6c^2_{k_1}c^2_{k_2}$'),
    Term(z_groups=(3,1)),
    Term(z_groups=(4,), expectation='$c^4_k$'),
    None,
    # with alpha
    Term(alpha=1, z_groups=(1,1,1)),
    Term(alpha=1, z_groups=(2,1)),
    Term(alpha=1, z_groups=(3,)),
    Term(alpha=2, z_groups=(1,1)),
    Term(alpha=2, z_groups=(2,), expectation=r'$6\alpha^2c_k^2$'),
    Term(alpha=3, z_groups=(1,)),
    Term(alpha=4, expectation=r'$\alpha^4$'),
]
labels1 = [r"$\alpha$", r"$\beta$", r"$Z_kc_k$", r"$E$"]


terms2=[
    #with beta
    Term(beta=1, alpha=0, z_groups=(1,1,1)),
    Term(beta=1, alpha=0, z_groups=(2,1)),
    Term(beta=1, alpha=0, z_groups=(3,)),
    Term(beta=1, alpha=1, z_groups=(1,1)),
    Term(beta=1, alpha=1, z_groups=(2,), delta_beta=r'$2\!\cdot\!12\alpha\beta c_k^2$'),
    Term(beta=1, alpha=2, z_groups=(1,)),
    Term(beta=1, alpha=3, delta_beta=r'$2\!\cdot\!4\alpha^3\beta$'),
    Term(beta=2, alpha=0, z_groups=(1,1)),
    Term(beta=2, alpha=0, z_groups=(2,)),
    Term(beta=2, alpha=1, z_groups=(1,)),
    Term(beta=2, alpha=2),
    Term(beta=3, alpha=0, z_groups=(1,)),
    Term(beta=3, alpha=1, z_groups=(), delta_beta=r'$2\!\cdot\!4\alpha\beta^3$'),
    Term(beta=4, alpha=0, z_groups=())
]
labels2 = [r"$\alpha$", r"$\beta$", r"$Z_kc_k$", r"$\Delta_\beta E$"]


# ----------------------------------------------------------------------
# MathText -> pygame
# ----------------------------------------------------------------------

_math_cache = {}

def draw_overbrace(screen: pygame.Surface, rect: pygame.Rect, y: int,
    width: int | None = None, height: int = 8, linewidth: int = 2, color="black"):
    """
    Draw a simple overbrace centered over rect.

             _____/\_____
            /            \

    The brace identifies the column belonging to an oversized formula.
    """
    if width is None: width = int(rect.width * 0.8)

    cx = rect.centerx
    x0 = cx - width // 2
    x1 = cx + width // 2

    q = width // 4

    points = [
        (x0,     y + height),
        (x0 + 5, y),
        (cx - q, y),
        (cx - 5, y),
        (cx,     y - height // 2),
        (cx + 5, y),
        (cx + q, y),
        (x1 - 5, y),
        (x1,     y + height),
    ]

    pygame.draw.lines(screen, color, False, points, linewidth)

def math_surface(formula: str, fontsize: int = 20) -> pygame.Surface:
    """
    Render matplotlib MathText to a transparent pygame surface.
    """
    key = (formula, fontsize)
    if key in _math_cache:
        return _math_cache[key]

    fig = plt.figure(figsize=(0.01, 0.01), dpi=200)
    fig.patch.set_alpha(0.0)

    text = fig.text(0, 0, formula, fontsize=fontsize, ha="left", va="bottom")

    fig.canvas.draw()

    bbox = text.get_window_extent()
    width = max(1, int(bbox.width) + 4)
    height = max(1, int(bbox.height) + 4)

    fig.set_size_inches(width / fig.dpi, height / fig.dpi)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    buffer.seek(0)
    surface = pygame.image.load(buffer, "formula.png").convert_alpha()

    _math_cache[key] = surface
    return surface

# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

WIDTH = 1070
HEIGHT = 280

FONT_SIZE = 10
LEFT = 80
TOP = 20

GAP = 140
COL_WIDTH = 70
ROW_HEIGHT = 60

DOT_RADIUS = 5
DOT_DX = 13
DOT_DY = 13


def draw_centered_text(screen: pygame.Surface, font: pygame.font.Font, text: str, rect: pygame.Rect):
    surface = font.render(text, True, "black")
    screen.blit(surface, surface.get_rect(center=rect.center))

def draw_math_centered (screen: pygame.Surface, formula: str, rect: pygame.Rect, fontsize: int = FONT_SIZE):
    surface = math_surface(formula, fontsize)

    # Oversized formula: mark its associated column.
    if surface.get_width() > rect.width:
        formula_rect = surface.get_rect(center=(rect.center[0], rect.center[1]+6))
        brace_y = formula_rect.top-6
        draw_overbrace(screen, rect, y=brace_y, width=int(formula_rect.width))
    else: 
        formula_rect = surface.get_rect(center=rect.center)
    screen.blit(surface, formula_rect)

def cell_rect(x: int, row: int) -> pygame.Rect:
    return pygame.Rect(x, TOP + row * ROW_HEIGHT, COL_WIDTH, ROW_HEIGHT)


def draw_power_dots(screen: pygame.Surface, rect: pygame.Rect, count: int):
    """
    Draw repeated occurrences of one variable horizontally.
    """
    total_width = (count - 1) * DOT_DX
    x0 = rect.centerx - total_width / 2

    for i in range(count):
        pygame.draw.circle(screen, "black",
            (round(x0 + i * DOT_DX), rect.centery), DOT_RADIUS)


def draw_z_dots(screen: pygame.Surface, rect: pygame.Rect, groups: tuple[int, ...]):
    """
    Each tuple entry is one distinct Z_k.

    (2,)   -> • •
    (1,1)  -> •
              •

    (2,1)  -> • •
              •
    """
    if not groups:
        return

    total_height = (len(groups) - 1) * DOT_DY
    y0 = rect.centery - total_height / 2

    for row, multiplicity in enumerate(groups):
        total_width = (multiplicity - 1) * DOT_DX
        x0 = rect.centerx - total_width / 2
        y = y0 + row * DOT_DY

        for i in range(multiplicity):
            pygame.draw.circle(screen, "black", (round(x0 + i * DOT_DX), round(y)), DOT_RADIUS)


def draw_table(screen: pygame.Surface, terms:list[Term], labels:list[str]):
    screen.fill("white")

    font = pygame.font.SysFont("segoeui", 20, bold=True)
    small_font = pygame.font.SysFont("segoeui", 17)
    
    ncols = len(terms)

    table_width = ncols * COL_WIDTH
    table_height = len(labels) * ROW_HEIGHT

    # --------------------------------------------------------------
    # Grid
    # --------------------------------------------------------------

    line_color = (180, 180, 180)
    x = LEFT

    for term in terms:
        pygame.draw.line(screen, line_color, (x, TOP), (x, TOP + table_height), 1)
        if term is None: x+= GAP
        else:
            for r in range(len(labels)+1):
                y = TOP + r * ROW_HEIGHT
                pygame.draw.line(screen, line_color, (x, y), (x + COL_WIDTH, y), 1)
            x = x + COL_WIDTH
    pygame.draw.line(screen, line_color, (x, TOP), (x, TOP + table_height), 1)

    # --------------------------------------------------------------
    # Row labels
    # --------------------------------------------------------------

    for row, label in enumerate(labels):
        rect = pygame.Rect(0, TOP + row * ROW_HEIGHT, LEFT - 10, ROW_HEIGHT)
        draw_math_centered(screen, label, rect)

    # --------------------------------------------------------------
    # Terms
    # --------------------------------------------------------------
    x = LEFT

    for term in terms:
        if term is None:
            x += GAP
            continue

        for (row, label) in enumerate(labels):
            if label==r"$\alpha$": draw_power_dots(screen, cell_rect(x, row), term.alpha)
            elif label==r"$\beta$": draw_power_dots(screen, cell_rect(x, row), term.beta) 
            elif label==r"$Z_kc_k$": draw_z_dots(screen, cell_rect(x, row), term.z_groups)
            elif label==r"$E$": draw_math_centered(screen, term.expectation, cell_rect(x, row))
            elif label==r"$\Delta_\beta E$": draw_math_centered(screen, term.delta_beta, cell_rect(x, row))
            else: raise ValueError (f"Unknown row label {label}")
        x += COL_WIDTH

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fourth moment term types")
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
draw_table(screen, terms1, labels1)
pygame.image.save(screen, "visualization_frames/terms-1.png")
draw_table(screen, terms2, labels2)
pygame.image.save(screen, "visualization_frames/terms-2.png")
pygame.quit()
