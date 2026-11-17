#!/bin/bash
# Run on the cluster; paste output back. Finds ATAC / CTCF / RNA / WGBS / histone tracks and eig status.
B=/gpfs/Scratch/DovatLab/Big_project2026
echo "== eig files"; ls $B/analysis_hg38/compartments/eig 2>/dev/null | wc -l
echo "== cf_libs.tsv head + marks"; find $B -maxdepth 3 -name 'cf_libs.tsv' | head -3
f=$(find $B -maxdepth 3 -name 'cf_libs.tsv' | head -1); [ -n "$f" ] && { head -3 "$f"; echo; cut -f2 "$f" | sort | uniq -c | head -20; }
echo "== ATAC"; find $B -maxdepth 5 \( -iname '*atac*.bw' -o -iname '*atac*.bigwig' -o -iname '*atac*peak*' \) 2>/dev/null | head -15
echo "== CTCF"; find $B -maxdepth 5 \( -iname '*ctcf*.bw' -o -iname '*ctcf*.bigwig' -o -iname '*ctcf*peak*' -o -iname '*ctcf*.bed*' \) 2>/dev/null | head -15
echo "== WGBS"; find $B -maxdepth 5 \( -iname '*wgbs*' -o -iname '*methyl*bw' -o -iname '*.bedGraph.gz' \) 2>/dev/null | head -10
echo "== counts"; for p in atac ctcf h3k27ac h3k27me3 rna; do echo -n "$p: "; find $B -maxdepth 6 -iname "*$p*" 2>/dev/null | wc -l; done
echo "== genotype tables"; find $B -maxdepth 3 -iname '*genotype*' -o -maxdepth 3 -iname '*groups_7*' 2>/dev/null | head
