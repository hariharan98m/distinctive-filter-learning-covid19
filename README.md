# Learning Distinctive Conv filters for COVID-19 detection

**Contributors:** Hariharan, M., Karthik, R., Menaka, R.

### Highlights
- A novel filter optimization module that restricts maximal activations of neurons in a CNN layer to respond to particular classes of Pneumonia/COVID-19.
- Tested on a fused dataset curated from different sources.
- Find publication <a href="https://www.sciencedirect.com/science/article/pii/S1568494620306827">here</a>

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1568494620306827-fx2.jpg" />

COVID-19 is a deadly viral infection that has brought a significant threat to human lives. Automatic diagnosis of COVID-19 from medical imaging enables precise medication, helps to control community outbreak, and reinforces coronavirus testing methods in place. While there exist several challenges in manually inferring traces of this viral infection from X-ray, Convolutional Neural Network (CNN) can mine data patterns that capture subtle distinctions between infected and normal X-rays. To enable automated learning of such latent features, a custom CNN architecture has been proposed in this research. It learns unique convolutional filter patterns for each kind of pneumonia. This is achieved by restricting certain filters in a convolutional layer to maximally respond only to a particular class of pneumonia/COVID-19. The CNN architecture integrates different convolution types to aid better context for learning robust features and strengthen gradient flow between layers. The proposed work also visualizes regions of saliency on the X-ray that have had the most influence on CNN’s prediction outcome. To the best of our knowledge, this is the first attempt in deep learning to learn custom filters within a single convolutional layer for identifying specific pneumonia classes. Experimental results demonstrate that the proposed work has significant potential in augmenting current testing methods for COVID-19. It achieves an F1-score of 97.20% and an accuracy of 99.80% on the COVID-19 X-ray set.

**Citation**

<cite>Karthik, R., Menaka, R., & Hariharan, M. (2021). Learning distinctive filters for COVID-19 detection from chest X-ray using shuffled residual CNN. Applied Soft Computing, 99, 106744.</cite>