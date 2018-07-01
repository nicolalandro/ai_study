#!/usr/bin/env bash
for entry in "trees"/*
do
  outname="${entry%.*}.png"
  echo "$entry"
  echo "$outname"
  dot -Tpng "$entry" -o "$outname"
done

