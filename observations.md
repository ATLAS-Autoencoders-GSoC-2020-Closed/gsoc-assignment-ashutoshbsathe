# Observations

1. The 4 variable space represented in the train and test set is actually a (m, pt, eta, phi) space
2. Apprently this is called LorentzVector [\[1\]](https://root.cern.ch/doc/master/namespaceROOT_1_1Math.html#a6cea5921731c7ac99dea921fb188df31) and [\[2\]](https://root.cern.ch/doc/master/classTLorentzVector.html)
3. Apparently this is a standard notation for storing the data at CERN
4. `m` could be mass, `pt` (rho) could be density, `phi` could be angle (actually the data is so that the mean of this column is 0 and the data ranges from -pi to +pi, uniformly), `eta` could be energy. (the calorie conversion constant in Table B.1 [here](https://cds.cern.ch/record/1083415/files/978-3-540-49045-6_BookBackMatter.pdf))
5. Train and test distributions are nearly similar judging from mean and standard deviation.
6. Max of `pt` in traindata is less than max of `pt`  in testdata.
7. `m` and `pt` columns have very large values whereas `phi` and `eta` are small values. To have a sensisble model, we need all values in similar range.
8. `phi` and `eta` can be normalized by min-max scaling since `phi`, `eta` appear to be bounded from my reserach. 
9. `m` and `pt` are difficult. Since theoretically they could be unbounded (mass and density are unbounded). This is a difficult choice between standardization and min-max scaling. Standardization inherently implies that all the future data MUST lie in the current distribution. This means that anomaly detection becomes very easy, but would give less freedom to choose activation functions.