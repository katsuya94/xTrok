#!/usr/bin/env bash

python ink.py
convert out.gif -coalesce temp.gif
convert -size 100x100 temp.gif -scale 400% final.gif
