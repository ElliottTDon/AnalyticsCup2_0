# AnalyticsCup2_0

Private working repo for Analytics Cup defense analysis.

**Question:** How can tracking and contextual data help us better understand the way players and teams defend?
(team shape, spacing, pressure, compactness, marking, defensive decision-making)

## Open on mobile

Phone-friendly HTML (GitHub Pages): https://elliotttdon.github.io/AnalyticsCup2_0/

Start here on a phone:

- [Visual defensive EDA](https://elliotttdon.github.io/AnalyticsCup2_0/eda.html) — 15 charts
- [Review notes / brainstorm](https://elliotttdon.github.io/AnalyticsCup2_0/review.html)
- [Full EDA notebook](https://elliotttdon.github.io/AnalyticsCup2_0/eda_notebook.html)
- [Graph experiment notebook](https://elliotttdon.github.io/AnalyticsCup2_0/defend.html)

Local analysis lives under `analysis/`. Regenerate figures + Pages assets:

```bash
# expects SkillCorner open data at opendata/data/matches/
python analysis/eda_visual_defense.py
```

Tutorial notebooks below are copied from [SkillCorner open data](https://github.com/SkillCorner/opendata) with saved outputs. Credit: SkillCorner.

- [`notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part1_Visualization_with_SkillCorner_Tutorial.ipynb`](notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part1_Visualization_with_SkillCorner_Tutorial.ipynb)
- [`notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part2_Multiple_Metrics_and_Z_Scores_Tutorial.ipynb`](notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part2_Multiple_Metrics_and_Z_Scores_Tutorial.ipynb)
- [`notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part3_Building_Striker_Archetypes_Tutorial.ipynb`](notebooks/tutorials/01_Getting_Started_with_SkillCorner_Data/Part3_Building_Striker_Archetypes_Tutorial.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part1_Aggregating_Dynamic_Events_Tutorial.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part1_Aggregating_Dynamic_Events_Tutorial.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part2_Data_Aggregating_Phases_of_Play_Tutorial.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part2_Data_Aggregating_Phases_of_Play_Tutorial.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part3_Offball_Runs_Pitch_visualization.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part3_Offball_Runs_Pitch_visualization.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part4_Merging_Dynamic_Events_and_Tracking_Data_Tutorial.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part4_Merging_Dynamic_Events_and_Tracking_Data_Tutorial.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part5_Animated_2D_Video_From_Tracking_And_Events.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part5_Animated_2D_Video_From_Tracking_And_Events.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part6_BuildYourOwnMetric_Detecting_and_Evaluating_Cutback_Opportunities.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part6_BuildYourOwnMetric_Detecting_and_Evaluating_Cutback_Opportunities.ipynb)
- [`notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part7_Pitch_Maps_Tracking_by_Phase_Tutorial.ipynb`](notebooks/tutorials/02_Working_with_Game_Intelligence_and_Dynamic_Events/Part7_Pitch_Maps_Tracking_by_Phase_Tutorial.ipynb)
- [`notebooks/tutorials/03_Basics_of_Tracking/Open_Data_Getting_Started_with_Tracking_and_Kloppy_Tutorial.ipynb`](notebooks/tutorials/03_Basics_of_Tracking/Open_Data_Getting_Started_with_Tracking_and_Kloppy_Tutorial.ipynb)
- [`notebooks/tutorials/03_Basics_of_Tracking/Open_Data_Tracking_Tutorial.ipynb`](notebooks/tutorials/03_Basics_of_Tracking/Open_Data_Tracking_Tutorial.ipynb)
- [`notebooks/tutorials/04_Visualizations/OBR_Simple_Radar_Viz.ipynb`](notebooks/tutorials/04_Visualizations/OBR_Simple_Radar_Viz.ipynb)
- [`notebooks/tutorials/04_Visualizations/Sectioned_Summary_Table_Viz_Tutorial.ipynb`](notebooks/tutorials/04_Visualizations/Sectioned_Summary_Table_Viz_Tutorial.ipynb)
- [`notebooks/tutorials/05_Body_Pose/Part1_Getting_Started_with_Body_Pose.ipynb`](notebooks/tutorials/05_Body_Pose/Part1_Getting_Started_with_Body_Pose.ipynb)

Local EDA notes live under `analysis/`.
