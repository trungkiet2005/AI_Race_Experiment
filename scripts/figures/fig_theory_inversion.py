"""Inverting the evolutionary model through the observed rates compresses them.

These two panels used to be panels b and c of the main paper's theory figure.
The main text never argued the inversion; it argues one thing, that the model
is a step and the routes are a ramp, and that claim is the whole of the
main-paper panel now.  The inversion is a claim about the model's
identifiability, it belongs beside the sweep that is written out here, and a
main-paper float is not the place for a claim the main paper does not make.

The claim, and it is about resolution rather than about order.  Because the only
place the model bends is its cliff, asking which risk would make the model emit
an observed rate sends nearly every answer to the same place.  Twenty-four
route-by-risk cells, drawn from three configured levels that span 0.80 of the
risk axis, come back inside a band 0.12 wide.

Be precise about what that does and does not say, because an earlier wording
here claimed more than the panel draws.  The order survives: all eight
invertible routes return their three implied risks in the configured order, and
seven of the eight cells run at 0.1 land strictly left of every cell run at 0.9,
the two extreme rows sharing only 0.0006 of the axis.  What the inversion loses
is resolution, not ranking.  Adjacent rows do overlap, the 0.1 and 0.6 rows on
0.0147 and the 0.6 and 0.9 rows on 0.0129, and the configured range is
compressed about sevenfold.

Panels
  a  The inversion as a slope chart: configured risk on the left at full scale,
     the risk the model would need in order to emit the observed rate on the
     right at the same full scale.  The fan closes.
  b  That band magnified, one row per configured level.  Adjacent rows overlap
     and the whole configured span collapses to 0.12, so a rate read back
     through this model fixes an implied risk only to about a tenth of the axis
     it was configured on.  Where two markers would land on each other the
     second is nudged off its row with a leader back to it, and the panel says
     so, because an undeclared vertical offset is the one device this panel
     cannot afford.

Every number here is computed by ``fig_theory_versus_behaviour.compute``, the
same call the main-paper figure makes, so the two figures cannot drift apart.

What this figure does NOT show.  Nothing here is fitted.  The inversion runs at
the weak-selection point, which is the one that gives the model its best chance
of spreading the cells apart; at the reference strength the inverse is defined
only inside a 0.005-wide window and inverting there would manufacture the
compression.  It is not an estimate of anybody's perceived risk: it says what
the model would have to be handed, which is a statement about the model.  And a
compressed inverse is not an unresponsive route.  All nine routes move the right
way with risk, Claude Opus 5 most steeply of all; eight of them merely also land
inside the model's own range, which is what lets the inversion answer at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402
from fig_theory_versus_behaviour import (  # noqa: E402
    compute,
    panel_b,
    panel_c,
    report,
)


def draw(res: dict, d: dict) -> None:
    fig = plt.figure(figsize=(S.TEXT, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_b(ax_a, res, d, letter="a")
    panel_c(ax_b, res, d, letter="b")

    cells = res["implied"].size
    ax_b.annotate(
        f"{cells - d['refused']} of {cells} cells invert. The other "
        f"{d['refused']} are Claude Opus 5 on 0% or 100%, rates no risk in\n"
        "the model produces, so they are refused rather than placed.",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -27),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.MUTED, linespacing=1.35)

    S.save(fig, "theory_inversion", width=S.TEXT)


def main() -> None:
    res = compute()
    draw(res, report(res))


if __name__ == "__main__":
    main()
