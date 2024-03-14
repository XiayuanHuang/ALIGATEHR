# ALIGATEHR


ALIGATEHR is a generic framework for learning patient representations, 
which models predicted family relations using a graph attention network of recurrent neural network (RNN) nodes.
To further enhance the quality of the learned representations, 
we additionally integrate a medical ontology of diagnosis codes into the [attention mechanism]([https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1710.10903)).
ALIGATEHR is a interpretable model,
allowing us to quantitatively assess the impact of the health history of family members on the disease risk of a patient.
The pedigree-based attention mechanism enables ALIGATEHR to capture genetic aspects of diseases using only EHR data as input.
Our model, ALIGATEHR, aims to explicitly capture dependencies between related patients and diseases from EHRs and medical ontologies, 
to learn more informative patient representations that can be utilized for a variety of downstream tasks. ALIGATEHR, by design, 
focuses on the sequential order of visits for each patient, without considering the temporal intervals between visits.

#### Code Description

The  code trains an RNN to predict disease risk of future visit by incorporating family history information infused from first-degree relatives and medical ongology knowledge.
Two attention-based graphs are built wiht attention layer simultaneously for learning latent patient representaion for each visit. Enventually, the latent patient representation
of each visit is inputted into RNN for disease risk prediction.


#### ALIGATEHR Framework
ALIGATEHR consists of two parallel processes: 1) Construction of pedigree graph: the patient representation h' aggregates health information from relatives; and 2) Construction of ontology graph: the representation g captures dependencies between diseases. The final representation v_t merges both patient and ontology information to represent a patient’s disease status for each visit. Finally, a series of visits is fed into a neural network model for the risk prediction task. ![Illustration of ALIGATEHR](Figure1.pdf)
