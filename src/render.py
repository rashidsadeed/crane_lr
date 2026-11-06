"""Pygame renderer for CraneEnv: side view of the gantry, trolley, cable, container and stacks.

Used through the standard Gymnasium API:
    env = CraneEnv(render_mode="human")      # opens a window, plays each step in real time
    env = CraneEnv(render_mode="rgb_array")  # env.render() returns the frame as an image array

In "human" mode every 0.01 s substep is drawn, so the motion is smooth (100 frames/s).
"""
import math

import numpy as np
import pygame

import dynamics as dyn
import geometry as geo

# colours (RGB)
BG = (250, 249, 245)
INK = (44, 44, 42)
QUIET = (136, 135, 128)
STEEL = (95, 94, 90)
STACK = (216, 90, 48)
STACK_EDGE = (113, 43, 19)
CONTAINER = (127, 119, 221)
CONTAINER_EDGE = (60, 52, 137)
TARGET = (29, 158, 117)
GOOD = (15, 110, 86)
BAD = (163, 45, 45)

OUTCOME_TEXT = {
    "success": ("SUCCESS", GOOD),
    "bad_landing": ("BAD LANDING", BAD),
    "collision": ("COLLISION WITH A STACK", BAD),
    "floor_collision": ("SET DOWN AWAY FROM THE TARGET", BAD),
    "end_stop": ("TROLLEY HIT THE END BUFFER", BAD),
    "hard_top_stop": ("HOIST HIT THE TOP STOP TOO FAST", BAD),
    "timeout": ("TIMEOUT", BAD),
}


class CraneRenderer:
    def __init__(self, params, mode: str, width: int = 1000, speed: float = 1.0):
        self.p = params
        self.mode = mode
        self.speed = speed                      # 2.0 = twice real time
        # gantry legs stand outside the area the container can reach (rail end + w_c/2 + 0.3)
        self.leg = params.w_c / 2 + 0.3
        # world window: legs plus a little room, z from -0.6 m to above the rail
        self.x0, self.x1 = params.x_min - self.leg - 0.5, params.x_max + self.leg + 0.5
        self.z0, self.z1 = -0.6, params.H + 1.4
        self.scale = width / (self.x1 - self.x0)          # pixels per metre
        height = int((self.z1 - self.z0) * self.scale) + 70                # + HUD strip
        self.size = (width, -(-height // 16) * 16)       # multiple of 16: what video codecs expect
        pygame.display.init()             # display and fonts only: no audio needed
        pygame.font.init()
        if mode == "human":
            pygame.display.set_caption("Overhead gantry crane")
            self.screen = pygame.display.set_mode(self.size)
        else:
            self.screen = pygame.Surface(self.size)
        self.font = pygame.font.Font(None, 24)
        self.big = pygame.font.Font(None, 40)
        self.clock = pygame.time.Clock()
        self.closed = False

    # ------------------------------------------------------------------ coordinates
    def px(self, x: float, z: float) -> tuple:
        """World (x, z) [m] -> screen pixels (z up -> y down)."""
        return (int(round((x - self.x0) * self.scale)),
                int(round((self.z1 - z) * self.scale)))

    def _dashed_rect(self, x_lo, x_hi, z_lo, z_hi, colour, dash=0.15):
        corners = [(x_lo, z_hi), (x_hi, z_hi), (x_hi, z_lo), (x_lo, z_lo), (x_lo, z_hi)]
        for (xa, za), (xb, zb) in zip(corners, corners[1:]):
            n = max(1, int(math.hypot(xb - xa, zb - za) / dash))
            for i in range(0, n, 2):
                t0, t1 = i / n, min(1.0, (i + 1) / n)
                pygame.draw.line(self.screen, colour,
                                 self.px(xa + (xb - xa) * t0, za + (zb - za) * t0),
                                 self.px(xa + (xb - xa) * t1, za + (zb - za) * t1), 2)

    # ------------------------------------------------------------------ drawing
    def draw_frame(self, env, s, t: float, outcome=None):
        p = self.p
        scr = self.screen
        scr.fill(BG)

        # floor, gantry beam and legs, rail-end buffers
        pygame.draw.line(scr, INK, self.px(self.x0, 0), self.px(self.x1, 0), 2)
        rail_z = p.H + 0.25
        x_l, x_r = p.x_min - self.leg, p.x_max + self.leg
        pygame.draw.rect(scr, STEEL, (*self.px(x_l, rail_z + 0.15),
                                      int((x_r - x_l) * self.scale), int(0.3 * self.scale)))
        for xb in (x_l, x_r):
            pygame.draw.line(scr, STEEL, self.px(xb, rail_z + 0.15), self.px(xb, 0), 4)
        for xb, sgn in ((p.x_min, -1), (p.x_max, 1)):        # end buffers at the trolley limits
            pygame.draw.rect(scr, STEEL, (*self.px(xb + sgn * 0.45 - 0.08, rail_z - 0.15),
                                          int(0.16 * self.scale), int(0.3 * self.scale)))

        # start position and target
        lay = env.layout
        half = p.w_c / 2
        pygame.draw.line(scr, QUIET, self.px(lay.x_A, -0.15), self.px(lay.x_A, -0.45), 2)
        scr.blit(self.font.render("A", True, QUIET), self.px(lay.x_A + 0.12, -0.1))
        self._dashed_rect(lay.x_B - half, lay.x_B + half, 0.0, p.h_c, TARGET)
        scr.blit(self.font.render("B", True, TARGET), self.px(lay.x_B + 0.12, -0.1))

        # stacks (one rectangle per container, so the stacking is visible)
        for st in lay.stacks:
            for k in range(st.n_high):
                r = pygame.Rect(*self.px(st.center - half, (k + 1) * p.h_c),
                                int(p.w_c * self.scale), int(p.h_c * self.scale))
                pygame.draw.rect(scr, STACK, r)
                pygame.draw.rect(scr, STACK_EDGE, r, 2)

        # trolley, cable, container
        x, _, l, _, theta, _ = s
        hook = dyn.hook_point(s, p)
        pygame.draw.aaline(scr, INK, self.px(x, p.H), self.px(*hook))
        pygame.draw.aaline(scr, INK, self.px(x + 0.03, p.H), self.px(hook[0] + 0.03, hook[1]))
        tw, th = 0.9, 0.35
        pygame.draw.rect(scr, INK, (*self.px(x - tw / 2, p.H + th), int(tw * self.scale),
                                    int(th * self.scale)), border_radius=3)
        corners = [self.px(cx, cz) for cx, cz in geo.container_corners(s, p)]
        pygame.draw.polygon(scr, CONTAINER, corners)
        pygame.draw.aalines(scr, CONTAINER_EDGE, True, corners)

        # heads-up display
        _, _, A = dyn.swing(s, p)
        z_b = p.H - l - p.h_c
        hud_y = self.size[1] - 60
        pygame.draw.line(scr, QUIET, (0, hud_y - 8), (self.size[0], hud_y - 8), 1)
        line1 = (f"t = {t:5.1f} s     layout {lay.kind}     x = {x:5.2f} m     "
                 f"height = {z_b:4.2f} m     swing = {100 * A:5.1f} cm     "
                 f"angle = {math.degrees(theta):+5.1f} deg")
        scr.blit(self.font.render(line1, True, INK), (12, hud_y))
        extra = getattr(env, "hud_text", "")
        if extra:
            scr.blit(self.font.render(extra, True, QUIET), (12, hud_y + 26))
        if outcome is not None:
            text, colour = OUTCOME_TEXT.get(outcome, (outcome.upper(), BAD))
            img = self.big.render(text, True, colour)
            scr.blit(img, (self.size[0] // 2 - img.get_width() // 2, 18))

    def _pump(self):
        """Keep the window responsive; closing it ends the viewer."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True

    def show(self, env, states, t_start: float, outcome=None):
        """'human' mode: draw each substep state at real speed."""
        dt = self.p.dt_sim
        for i, s in enumerate(states):
            if self.closed:
                return
            last = i == len(states) - 1
            self.draw_frame(env, s, t_start + (i + 1) * dt, outcome if last else None)
            self._pump()
            pygame.display.flip()
            self.clock.tick(self.speed / dt)
        if outcome is not None and not self.closed:     # hold the final frame briefly
            for _ in range(int(1.5 / dt)):
                self._pump()
                self.clock.tick(1 / dt)

    def frame(self, env, s, t: float, outcome=None) -> np.ndarray:
        """'rgb_array' mode: the frame as an (H, W, 3) uint8 array."""
        self.draw_frame(env, s, t, outcome)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        pygame.display.quit()
        pygame.font.quit()