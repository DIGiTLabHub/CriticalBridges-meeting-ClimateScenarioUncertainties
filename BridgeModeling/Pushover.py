#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openseespy.opensees as op
import pandas as pd
from pathlib import Path
from RecorderColFiber import *
from model_setup import build_model

# === Auto-load the latest Excel ===
result_dir = Path("RecorderData/results")
excel_path = max(result_dir.glob("Scour_Materials_20250425_142513 - Copy.xlsx"), key=lambda f: f.stat().st_mtime)
print(f"📂 Using Excel file: {excel_path.name}")

# === Scenario sheet mapping ===
scenario_sheets = {
    # "Missouri": "Scenario_1_Missouri_River"
    # ,
    # "Colorado": "Scenario_2_Colorado_River"
    # ,
    "Extreme": "Scenario_3_Extreme_Case"
}

# === User-defined input parameters ===
IDctrlNode = 5201
LCol = 13050.0
Weight = 28.703462  # MN

DmaxRatio  = 0.05
DincrRatio = 0.0001
maxNumIter = 100
tol        = 1.0e-6

IDctrlDOF   = 2
loadNodeTag = 5201
patternTag  = 200
load_vector = [0.0, Weight, 0.0, 0.0, 0.0, 0.0]

# === Derived ===
Dmax  = DmaxRatio * LCol
Dincr = DincrRatio * LCol
Nsteps = int(Dmax / Dincr)

# === Loop through each scenario ===
for label, sheet_name in scenario_sheets.items():
    print(f"\n🟦 Processing scenario: {label}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    for i, row in df.iterrows():
        scourDepth = row["Scour_Depth_mm"] / 1000.0
        scourDepthmm = round(row["Scour_Depth_mm"] + LCol, 1)
        fc = row["fc'_MPa"]
        fy = row["fy_MPa"]

        print(f"\n🔄 {label} | Sample {i+1}: Scour = {scourDepth:.3f} m | fc' = {fc:.2f} MPa | fy = {fy:.2f} MPa")

        # === 1. Build model ===
        op.wipe()
        build_model(fc, fy, scourDepthmm)

        # === 2. Gravity analysis ===
        op.constraints("Transformation")
        op.numberer("RCM")
        op.system("BandGeneral")
        op.algorithm("Newton")
        op.test("NormDispIncr", 1.0e-6, 1000)
        op.integrator("LoadControl", 1.0)
        op.analysis("Static")

        result = op.analyze(1)
        if result != 0:
            print(f"❌ Gravity failed for {label} sample {i+1}")
            continue
        op.reactions()
        op.loadConst("-time", 0.0)

        # === 3. Lateral load ===
        op.timeSeries('Linear', 2)
        op.pattern('Plain', patternTag, 2)
        op.load(loadNodeTag, *load_vector)

        # === 4. Recorders ===
        depth = round(row["Scour_Depth_mm"], 1)
        folder = f"RecorderData/{label}/scour_{depth:.1f}"
        define_recorders(folder=folder)
        define_displacement_recorders(folder=folder)

        # === 5. Analysis setup ===
        op.wipeAnalysis()
        op.constraints('Transformation')
        op.numberer('RCM')
        op.system('BandGeneral')
        op.test('EnergyIncr', tol, maxNumIter)
        op.algorithm('Newton')
        op.integrator('DisplacementControl', IDctrlNode, IDctrlDOF, Dincr)
        op.analysis('Static')

        ok = op.analyze(Nsteps)
        print(f"Initial result: {ok}")

        # === 6. Fallback if failed ===
        if ok != 0:
            test_dict = {
                1: 'NormDispIncr',
                2: 'RelativeEnergyIncr',
                4: 'RelativeNormUnbalance',
                5: 'RelativeNormDispIncr',
                6: 'NormUnbalance'
            }
            algo_dict = {
                1: 'KrylovNewton',
                2: 'SecantNewton',
                4: 'RaphsonNewton',
                5: 'PeriodicNewton',
                6: 'BFGS',
                7: 'Broyden',
                8: 'NewtonLineSearch'
            }

            for test_type in test_dict.values():
                for algo_type in algo_dict.values():
                    if ok != 0:
                        if algo_type in ['KrylovNewton', 'SecantNewton']:
                            op.algorithm(algo_type, '-initial')
                        else:
                            op.algorithm(algo_type)
                        op.test(test_type, tol, 1000)
                        ok = op.analyze(Nsteps)
                        print(f"Trying {test_type} + {algo_type} → Result: {ok}")
                        if ok == 0:
                            break
                if ok == 0:
                    break

        # === 7. Final results ===
        try:
            u = op.nodeDisp(IDctrlNode, IDctrlDOF)
            print(f"✅ Final uy @ Node {IDctrlNode}: u = {u:.6f} m")
        except:
            print("❌ Displacement retrieval failed.")

        print("--------------------------------------------------")

print("✅ All scenarios processed.")


# In[ ]:




