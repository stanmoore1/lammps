#!/bin/bash
# Compare the thermo output of the two memory models, input by input.
#
# This is the one check that does not go through the sync protocol: it looks at
# the numbers the run produced, so it catches a wrong value that every detector
# agreed was properly synced.  That only works if the numbers are actually
# extracted, so the two thermo formats are both handled and an input that
# yields nothing is reported as UNCOMPARED rather than counted as agreement --
# 26 inputs use "thermo_style multi", and an extractor that silently returns
# empty for both sides makes every one of them look like a pass.
#
# The wall-clock columns are dropped for the same reason compare-thermo.py drops
# them: they differ on every pair of runs, so an input whose thermo_style names
# CPU or S/CPU otherwise reports as a divergence and is not one.  Three findings
# in FINDINGS.md were exactly that.
SP=/tmp/claude-0/-home-user-lammps/e39b99de-89e3-50b6-a67f-c617e8ffcc75/scratchpad/sync
thermo() {
  zcat "$1" 2>/dev/null | awk '
    function is_timing(h) {
      return (h == "CPU" || h == "S/CPU" || h == "T/CPU" || h == "CPULeft" ||
              h == "CPUleft" || h == "Elapsed" || h == "WallTime")
    }
    # standard style: a "Step ..." header, then rows, until Loop time.  Remember
    # which columns the header named so the timing ones can be left out of every
    # row below -- they differ on every run and mean nothing here.
    /^ *Step / {
      inblk=1; ncol=$NF=="" ? NF-1 : NF
      for (i=1; i<=NF; i++) keep[i] = !is_timing($i)
      ncol=NF; next
    }
    /^Loop time/ { inblk=0; next }
    # multi style: "------------ Step N ----- CPU = ..." then named values
    /^-+ Step +[0-9]+ +-+ CPU/ { print "STEP " $3; next }
    /^(TotEng|PotEng|E_bond|E_angle|E_dihed|E_impro|E_vdwl|E_coul|E_long|Press|Temp|KinEng|Volume) /  { print; next }
    inblk && /^ *[-0-9]/ {
      if (NF != ncol) next            # a partial row, not a thermo line
      out=""
      for (i=1; i<=NF; i++) if (keep[i]) out = out (out=="" ? "" : " ") $i
      print out; next
    }
  ' | grep -vE "WARNING|MPI task|CPU ="
}
n=0; d=0; u=0
while read -r f; do
  name=$(echo "$f" | sed 's|examples/||; s|/|_|g')
  a=$SP/plain845/$name.out.gz; b=$SP/split845/$name.out.gz
  [ -f "$a" ] && [ -f "$b" ] || continue
  ta=$(thermo "$a"); tb=$(thermo "$b")
  if [ -z "$ta" ] && [ -z "$tb" ]; then
    u=$((u+1)); echo "UNCOMPARED $f (no thermo extracted from either side)"
    continue
  fi
  n=$((n+1))
  if [ "$ta" != "$tb" ]; then
    d=$((d+1))
    echo "DIVERGES $f"
    diff <(printf '%s\n' "$ta") <(printf '%s\n' "$tb") | head -6 | sed 's/^/    /'
  fi
done < $SP/all-inputs.txt
echo
echo "compared $n inputs, $d diverge, $u uncompared"
