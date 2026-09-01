Preset recipes (shortcuts)
==========================

Each preset is just a **pre-filled point** along the four axes (:doc:`composing`): a fixed *space*, *construction*, *method*, and *output* chosen to match a published method.  ``TDAPipeline.from_paper("<name>")`` returns the same kind of pipeline you would assemble by hand — see :doc:`composing` for how to build your own instead.

One runnable notebook per preset below, rendered with its **stored outputs** (the build never re-executes).  Each trains a small model on the laptop GPU (or generates data), runs the preset, and shows the paper's headline result.  See ``paper_reproduce/README.md`` for dataset notes and the model-first vs. explicit-extraction guide.

.. toctree::
   :maxdepth: 1

   notebooks/andreeva2024
   notebooks/ballester2024
   notebooks/birdal2021
   notebooks/dupuis2023
   notebooks/gabella2021
   notebooks/gabrielsson2019
   notebooks/gebhart2019
   notebooks/karuppiah2025
   notebooks/lacombe2021
   notebooks/liu2023
   notebooks/naitzat2020
   notebooks/ong2026
   notebooks/ramamurthy2019
   notebooks/rathore2021
   notebooks/rieck2019
   notebooks/ruppik2025
   notebooks/watanabe2021
   notebooks/zhou2023
