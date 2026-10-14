# LX521.4 top-baffle print pack

Start with [the build guide](docs/BUILD_GUIDE.md), then choose exact jobs from
[the file guide and estimates](to_print/FILE_GUIDE.md).

This pack contains actual STL/3MF files and adjacent orientation/plate
records. There are 42 choices: 68 sliced projects and eight GUI projects
across alternate lanes. Print a selected configuration, not every file.
PETG-GF `_GUI.3mf` projects require slicing and auditing in Bambu Studio.

The [qualification fixtures and procedure](docs/PETG_GF_QUALIFICATION.md)
are included. Obi-Wan remains a candidate: physical fit, print-process,
loaded and acoustic measurements are pending. Rendered images are CAD
illustrations, not photos of printed results.

Verify the downloaded archive with `python verify_package.py <archive.zip>`
and compare its external `.sha256` file. `SHA256SUMS.json` identifies the
individual payload files. Keep STL orientation records beside their meshes.

This is a print pack, not the complete source/CAD build repository. Some
engineering-reference links in the detailed docs require that repository.
The GUI slice audit command likewise requires its code, pinned Bambu
profiles and G-code validator. No printer is contacted by verification.
