---
name: Future works
about: Suggest an idea for this project
title: Planning
labels: enhancement
assignees: elaheoveisi

---

**Is your feature request related to a problem? Please describe.**
We need to use LLM to extract additional human factors features to improve the accuracy of the Random Forest model. Generally, providing more features enables the model to perform better.
**Describe the solution you'd like to explain**

I want to extract new "Human Factors" features from the narrative using a language model. These features will help enrich our dataset, which will be used in a Random Forest model to predict violations and errors. I will review some reports to identify human factor features that may not be explicitly mentioned in our existing dataset columns. We can extract these features from the narratives using the language model, creating a new dataset that includes all identified features. Then, we will incorporate these features into the available Random Forest model to enhance its predictive capabilities.
**Describe alternatives you've considered**
 Using ToT in LLM extraction to measure the LLM confidence and compete with RF.

**Additional context**
Our objective in enhancing features is to leverage LLM effectively, enabling us to predict accidents based on sound mathematical and statistical principles.

Code*
I will add new features extracted from the reports and run the LLM on the ASRS dataset to extract them, and feed them to the RF to measure its performance.
