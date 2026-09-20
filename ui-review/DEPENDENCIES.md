# Dependencies

None added. The page still loads three.js r128 from cdnjs on the first Map visit and Barlow from
Google Fonts, as before; the font request now also asks for Barlow Semi Condensed 600 (same family,
same host). The orbit controller, the fat intruder path and the screen-space labels are written in
`pitch/ui/scene3d.js` rather than pulled from three.js examples. Screenshots use the Chrome already
installed on this machine, so Playwright was not needed. `ui-review/tools/` uses the repo's existing
Python virtualenv (PIL is already in it).
