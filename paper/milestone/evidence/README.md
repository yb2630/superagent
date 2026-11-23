# Before/after PDF evidence

- `before-page{1,2}.png` are renders of `paper/milestone/main.pdf` at the base commit
  (`c09145f`, `main`), i.e. the milestone with the illustrative pilot table.
- `after-page{1,2}.png` are renders of `paper/milestone/main.pdf` at this branch head.

Both were rendered at 2x with `pypdfium2` from the committed PDF; page 3 is the reference list
and is omitted. The paper is compiled from `main.tex` with Tectonic 0.17.0 (XeTeX).
