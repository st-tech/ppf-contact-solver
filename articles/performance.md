## 🚀 How We Made the Solver 2 to 3 Times Faster

The solver is now 2 to 3 times faster than the previous version (commit 97ac227, May 21, 2026). Running the same scenes on the same machine, every scene got faster: on average about **2.9x per solver step** and **2.2x per finished frame**, and up to **5.8x** on the lightest scenes.

### 🧪 Method

- Same scenes and settings for both versions; only the solver differed.
- One NVIDIA L40S, scenes run one at a time so nothing competed for the card.
- "Per step" is one internal calculation. "Per frame" is one frame of the animation.

### 📊 Results

| Scene | Faster per step | Faster per frame |
|:--|:--:|:--:|
| [cards](../examples/cards.ipynb) | 5.80x | 5.65x |
| [yarn](../examples/yarn.ipynb) | 4.19x | 2.41x |
| [belt](../examples/belt.ipynb) | 4.00x | 3.69x |
| [friction](../examples/friction.ipynb) | 3.98x | 3.70x |
| [domino](../examples/domino.ipynb) | 3.45x | 3.06x |
| [trampoline](../examples/trampoline.ipynb) | 3.38x | 2.63x |
| [fitting](../examples/fitting.ipynb) | 3.21x | 2.08x |
| [woven](../examples/woven.ipynb) | 3.16x | 1.96x |
| [stack](../examples/stack.ipynb) | 3.13x | 2.51x |
| [noodle](../examples/noodle.ipynb) | 3.01x | 1.67x |
| [codim](../examples/codim.ipynb) | 2.89x | 1.92x |
| [drape](../examples/drape.ipynb) | 2.82x | 2.75x |
| [hang](../examples/hang.ipynb) | 2.72x | 1.67x |
| [roller](../examples/roller.ipynb) | 2.42x | 2.40x |
| [fishingknot](../examples/fishingknot.ipynb) | 2.41x | 1.44x |
| [trapped](../examples/trapped.ipynb) | 2.32x | 1.92x |
| [five-twist](../examples/five-twist.ipynb) | 2.32x | 1.97x |
| [ribbon](../examples/ribbon.ipynb) | 2.29x | 1.51x |
| [twist](../examples/twist.ipynb) | 2.27x | 2.11x |
| [curtain](../examples/curtain.ipynb) | 2.20x | 1.10x |
| [needle](../examples/needle.ipynb) | 2.02x | 1.85x |

Average: about 2.9x per step, 2.2x per frame. Nothing got slower.
