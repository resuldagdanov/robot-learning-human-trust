# Robot Learning of Human Trust
The codebase for the paper "Improving Trust Estimation in Human-Robot Collaboration Using Beta Reputation at Fine-grained Timescales", published in the 2025 [IEEE Robotics and Automation Letters (RA-L)](https://www.ieee-ras.org/publications/ra-l).

---
## [Paper](https://arxiv.org/abs/2411.01866) | [Code](https://github.com/resuldagdanov/robot-learning-human-trust) | [ArXiv](https://arxiv.org/abs/2411.01866) | [Slide](https://github.com/resuldagdanov/robot-learning-human-trust/tree/main/presentation/slides) | [Video](https://youtu.be/tSxQLM4Hr8s) | [Presentation](https://youtu.be/tSxQLM4Hr8s) | <a href="src/README.md">ReadMe</a>

---
## Authors
[Resul Dagdanov](https://profiles.uts.edu.au/Resul.Dagdanov), [Milan Andrejevic](https://profiles.uts.edu.au/Milan.Andrejevic), [Dikai Liu](https://profiles.uts.edu.au/Dikai.Liu), [Chin-Teng Lin](https://profiles.uts.edu.au/Chin-Teng.Lin)

## Watch YouTube Video
[![Watch Video](https://img.youtube.com/vi/tSxQLM4Hr8s/maxresdefault.jpg)](https://youtu.be/tSxQLM4Hr8s)
<figure>
    <figcaption style="text-align: center; font-style: italic;">Video 1: Detailed Explanation of the Proposed Framework <a href="https://youtu.be/tSxQLM4Hr8s">[YouTube Link]</a>
    </figcaption>
</figure>

---
## Follow <a href="src/README.md">ReadMe</a> File for Experiments and Source Code

---
## Citation
```bibtex
@misc{dagdanov2024trust,
      title={Improving Trust Estimation in Human-Robot Collaboration Using Beta Reputation at Fine-grained Timescales}, 
      author={Resul Dagdanov and Milan Andrejevic and Dikai Liu and Chin-Teng Lin},
      year={2024},
      eprint={2411.01866},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.01866}, 
}
```

---
## Teaser
<figure>
    <p align="center">
        <img src="presentation/images/teaser.png" width="640px" alt="Teaser"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 1: Teaser of the Proposed Framework</figcaption>
</figure>

---
## General Framework
<figure>
    <p align="center">
        <img src="presentation/images/framework.png" width="1080px" alt="Framework"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 2: General Framework</figcaption>
</figure>

---
## Autonomous Tiling with Collaborative Robot
<figure>
    <p align="center">
        <img src="presentation/videos/Cobot_Autonomous_Tiling_Operation_GIF.gif" width="640px" alt="Tiling Operation"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Video 2: Robot Executing Tiling Operation Autonomously after Learning from Demonstrations</figcaption>
</figure>


---
## Human Trust Measurement
<figure>
    <p align="center">
        <img src="presentation/images/trust_measurement.png" width="640px" alt="Measurement"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 3: Measurement of Human Trust Toward a Robot (7-point Likert Scale)</figcaption>
</figure>

---
## Methodology
<figure>
    <p align="center">
        <img src="presentation/images/methodology.jpg" width="640px" alt="Methodology"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 4: Illustration of an Iterative Human Trust Modeling Process (Proposed Framework)</figcaption>
</figure>

---
## Data Collection Process
<figure>
    <p align="center">
        <img src="presentation/images/human_interaction.PNG" width="640px" alt="Human Interaction"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Figure 5: Data Collection by Human Demonstrator in <a href="https://www.uts.edu.au/research/robotics-institute">UTS Robotics Institute</a> Lab Environment</figcaption>
</figure>

---
## Reward Function with Maximum-Entropy Optimization
<figure>
    <p align="center">
        <img src="presentation/videos/Simulation_Video_ROS.gif" width="640px" alt="Simulation Video"/>
    </p>
    <figcaption style="text-align: center; font-style: italic;">Video 3: Visualization of Robot Decision-Making Policy in ROS Simulation Environment during IRL Optimization</figcaption>
</figure>
