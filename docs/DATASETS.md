# Datasets and data governance

LaserPerception does not redistribute SemanticKITTI, KITTI, DALES, or future datasets. Data must
remain outside Git and be addressed through configuration or environment variables.

## SemanticKITTI source

Obtain KITTI odometry scans and SemanticKITTI labels from their official sources. Users are
responsible for the current terms and citations of both projects.

- Format and terms: <https://semantic-kitti.org/dataset.html>
- Official development kit: <https://github.com/PRBonn/semantic-kitti-api>
- Suggested variable: `LASERPERCEPTION_SEMANTICKITTI_ROOT`

The loader preserves remission; Experiment 001 excludes it from model features.

## DALES target

Obtain DALES through its official distribution path and review its current terms. Apache-2.0 does
not apply to DALES.

- Paper and class specification: <https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Varney_DALES_A_Large-Scale_Aerial_LiDAR_Data_Set_for_Semantic_Segmentation_CVPRW_2020_paper.html>
- Suggested variable: `LASERPERCEPTION_DALES_ROOT`

## Mapping provenance

SemanticKITTI IDs come from official `semantic-kitti.yaml`. The DALES paper defines unknown `0`,
ground `1`, vegetation `2`, cars `3`, trucks `4`, power lines `5`, fences `6`, poles `7`, and
buildings `8`. Grouping those verified IDs into six classes is LaserPerception's Experiment 001
policy and is tested in `ontology/mappings.py`.

Synthetic tests generate temporary data and need no public dataset download.
