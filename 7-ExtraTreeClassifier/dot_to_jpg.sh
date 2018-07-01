#!/usr/bin/env bash
#dot -Tpng "trees/0-tree.dot" -o "trees_images/0-tree.png"
i=0
for entry in "trees"/*
do
  outname="${entry%.*}.png"
  echo "$entry"
  echo "$outname"
  dot -Tpng "$entry" -o "$outname"
done

