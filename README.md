# network-slice-availability-prediction---ACM-SAC-2025

# 📌 Project Overview

This repository contains the code, data, and implementation for the paper:

**Title:** *Towards Efficient Mobility Management in 5G-Advanced: A Predictive Model for Network Slice Availability*  
**Authors:** Muhammad Ashar Tariq, Malik Muhammad Saad, Mahnoor Ajmal, Ayesha Siddiqa, Dongkyun Kim  
**Presented at:** ACM SAC 2025 (Symposium on Applied Computing)

---

In 5G-Advanced networks, the **unavailability of network slices** across certain Tracking Areas (TAs) within a Registration Area (RA) leads to:
- Increased **Mobility Registration Updates (MRUs)**
- Higher **signaling overhead**
- Risk of **service denial** when UE enters a TA without the requested slice

This project proposes a **predictive solution** using:
- 🔮 An **LSTM-based model** to predict future slice availability in the UE's next TA
- 🧠 A **mobility optimization algorithm** (future extension using Deep Q-Learning) to proactively manage UE movement and avoid service disruption

By anticipating slice availability before UE movement, the system enables **intelligent, low-latency, and uninterrupted mobility management** for future 5G deployments.

---

## 🧠 Core Contributions

- 🔍 Identifies the challenge of slice unavailability across TAs, leading to service denial and excessive MRUs.
- 🔮 Proposes an LSTM-based model to predict slice availability in the UE’s next TA.
- 🤖 Lays the foundation for reinforcement learning-based UE mobility optimization (future work).
- 🧪 Builds a simulation to compare predictive vs. baseline strategies on metrics like MRUs, service rejection, and slice access time.

---

## 🔮 Slice Availability Prediction (LSTM)

This project predicts slice availability in a UE's next Tracking Area (TA) to enable smarter mobility decisions in 5G-Advanced networks.

### 🔍 Slice Availability Prediction (LSTM)
A multi-label LSTM model forecasts the availability of each slice using:
- Historical slice presence
- UE mobility level
- Current & next TA (one-hot encoded)
- Time context (hour, weekday)

---

## 📊 Results & Evaluation

The LSTM model was evaluated on a 14-day time series dataset of slice availability across 10 TAs and 6 slices.

### ✅ Key Outcomes:
- Achieved stable multi-label prediction accuracy across slices
- Training/validation loss converged within 30 epochs
- Prediction plots closely matched actual slice availability trends

The simulation also showed that predictive mobility strategies (planned) can reduce:
- Service rejection when entering new TAs
- MRU occurrences due to unavailable slices

📄 For detailed results, loss graphs, and prediction comparisons, please refer to the published paper.

---

## 📚 Citation

If you use this code, dataset, or findings in your work, please cite the following paper:
> https://dl.acm.org/doi/abs/10.1145/3672608.3707836 (https://doi.org/10.1145/3672608.3707836)

```bibtex
@inproceedings{tariq2025towards,
  title={Towards Efficient Mobility Management in 5G-Advanced: A Predictive Model for Network Slice Availability},
  author={Tariq, Muhammad Ashar and Ajmal, Mahnoor and Saad, Malik Muhammad and Siddiqa, Ayesha and Park, Seri and Kim, Dongkyun},
  booktitle={Proceedings of the 40th ACM/SIGAPP Symposium on Applied Computing},
  pages={2040--2047},
  year={2025}
}
```

---

## 📬 Contact

For questions, collaboration, or additional details, feel free to reach out:

- 📧 Muhammad Ashar Tariq — [tariqashar5@gmail.com](mailto:tariqashar5@gmail.com)
- 🔗 LinkedIn — [linkedin.com/in/tariqashar5](https://www.linkedin.com/in/tariqashar5)


---

## 🔒 License

This repository is intended for **academic and research purposes only**.  
For commercial use or redistribution, please contact the authors.

---

