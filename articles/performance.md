## 🚀 How We Made the Solver 2 to 3 Times Faster

The solver is now 2 to 3 times faster per finished frame than the previous
version (commit 97ac227, May 21, 2026). Running the same scenes on the same
machine, every scene got faster: on average about **2.9x per finished frame**,
with a median of **2.2x**, and up to **5.4x** on the scenes that benefit most.

### 🧪 Method

- Same scenes and settings for both versions; only the solver differed.
- One NVIDIA L40S, scenes run one at a time so nothing competed for the card.
- "Per frame" is the wall-clock time to finish one frame of the animation. This
  is the number that matters, because it is what you actually wait for.

### 📊 Results

| Scene | Faster per frame |
|:--|:--:|
| [trampoline](../examples/trampoline.ipynb) | 5.40x |
| [drape](../examples/drape.ipynb) | 5.16x |
| [cards](../examples/cards.ipynb) | 4.82x |
| [friction](../examples/friction.ipynb) | 3.75x |
| [belt](../examples/belt.ipynb) | 3.65x |
| [hang](../examples/hang.ipynb) | 3.62x |
| [yarn](../examples/yarn.ipynb) | 3.23x |
| [domino](../examples/domino.ipynb) | 3.17x |
| [roller](../examples/roller.ipynb) | 2.55x |
| [woven](../examples/woven.ipynb) | 2.50x |
| [fishingknot](../examples/fishingknot.ipynb) | 2.23x |
| [stack](../examples/stack.ipynb) | 2.23x |
| [needle](../examples/needle.ipynb) | 2.14x |
| [curtain](../examples/curtain.ipynb) | 2.13x |
| [fitting](../examples/fitting.ipynb) | 2.11x |
| [ribbon](../examples/ribbon.ipynb) | 2.06x |
| [twist](../examples/twist.ipynb) | 2.03x |
| [five-twist](../examples/five-twist.ipynb) | 2.01x |
| [trapped](../examples/trapped.ipynb) | 1.94x |
| [noodle](../examples/noodle.ipynb) | 1.91x |
| [codim](../examples/codim.ipynb) | 1.73x |

Average: about 2.9x per finished frame, with a median of 2.2x. Nothing got slower.
