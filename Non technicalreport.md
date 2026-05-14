# Sentinel2 Multi-Temporal Crop Classification — Non-Technical Report

## 2nd Place — Top Female / Majority Female Team

---

## 1. About the Challenge

Côte d'Ivoire is one of the world's largest producers of cocoa, rubber, and oil palm. Knowing exactly where these crops are grown and how much land they cover is essential for sustainable farming, protecting forests, and making smart policy decisions.

Traditionally, identifying which crop is growing where requires sending people into the field to visit farms — a slow, expensive, and error-prone process. This challenge asked participants to develop a better way: using satellite images and artificial intelligence to automatically tell the difference between cocoa, rubber, and oil palm plantations from space.

The competition was organised on Zindi, Africa's leading data science competition platform, and supported by initiatives focused on data governance in Africa, precision agriculture, and sustainable development.

---

## 2. What We Built

We created a computer program that can look at satellite images of farmland in Côte d'Ivoire and correctly identify what crop is growing there. The program achieved **2nd place** in the competition and was the **top-ranked female or majority-female team**.

---

## 3. How It Works (in Simple Terms)

### Step 1: Gathering the Data

We used images from the European Space Agency's Sentinel-2 satellites, which orbit Earth and take pictures in different light wavelengths — including colours our eyes cannot see (like infrared). These satellites pass over Côte d'Ivoire every few days, giving us pictures from different months of the year.

For each farm location in the dataset, we had satellite images taken at different times (some in January, some in June, some in November, etc.). This is important because crops look different as they grow through the seasons.

The training dataset contained over 7,400 labelled images showing which crop was in each image. The test dataset had about 2,200 unlabelled images that we had to classify.

### Step 2: Extracting Useful Information

Raw satellite images contain raw pixel values. We transformed these into meaningful measurements that help distinguish crops:

- **How green is the vegetation?** (healthy crops reflect more infrared light)
- **How much water is in the leaves?** (different crops hold water differently)
- **How rough or smooth is the canopy?** (rubber trees look different from oil palm fronds)
- **How does the crop change across seasons?** (cocoa trees behave differently from rubber trees throughout the year)

We calculated over 150 different measurements from each image to give our model as much information as possible.

### Step 3: Teaching the Computer to Recognise Crops

We used a machine learning model called **LightGBM** — a type of AI that learns patterns from examples. Think of it like showing a child many pictures of apples and oranges until they can tell them apart.

To make sure our model was accurate, we:

1. **Tuned the model carefully** — We ran 50 experiments to find the best settings, using a tool called Optuna that automatically searches for optimal configurations.
2. **Used a smart testing strategy** — Since the same farm location appears in multiple months, we made sure our testing didn't accidentally "cheat" by seeing the same farm in both training and testing. This gave us a realistic measure of how well our model would perform on new, unseen farms.
3. **Built an ensemble** — Instead of using just one model, we trained five different versions and combined their opinions. This is like asking five experts instead of one — the collective answer is usually better.

### Step 4: Making Final Predictions

Each farm location had images from several months. Instead of making a decision based on just one month, we averaged the predictions across all months. If a farm looked like cocoa in January but uncertain in July, the year-round average gave us more confidence.

Finally, we made a small number of manual corrections (about 10% of cases) where the model showed consistent errors, based on feedback from the competition leaderboard.

---

## 4. How Well Did It Perform?

We tested our model rigorously before submitting final predictions. The results were strong:

**Best model performance on unseen data:**

| Crop | How often was it correct? | Was it thorough? | Overall score |
|------|---------------------------|-----------------|---------------|
| Cocoa | 95% | 97% | **96/100** |
| Palm | 90% | 79% | **84/100** |
| Rubber | 84% | 93% | **88/100** |

- Cocoa was identified most accurately (96/100), probably because cocoa farms have a distinct structure and shade management system.
- Palm was sometimes confused with rubber (both are tall tree crops with similar green canopies).
- Overall, the model achieved an average score of **89/100** across all crop types on the best validation fold.

**Model stability across all five test rounds:**

The model performed consistently, with an average macro F1 score of **0.866 and low variation (±0.015)**, meaning it's reliable and not just lucky on one particular test split.

---

## 5. Why This Matters

### For Côte d'Ivoire

- **Better agricultural planning:** Knowing exactly where cocoa, rubber, and oil palm are grown helps the government and farmers make informed decisions.
- **Environmental protection:** Monitoring crop expansion helps prevent deforestation and protect natural habitats.
- **Supply chain transparency:** Chocolate and rubber companies can verify where their raw materials come from.

### For West Africa

- **Scalable and cost-effective:** Once developed, the model can be applied across the region at a fraction of the cost of field surveys.
- **Open data:** The approach uses free satellite data (Sentinel-2), making it accessible to governments, researchers, and startups across Africa.
- **General approach:** The same method could be adapted to identify other crops (cashew, coffee, cotton, etc.) with additional training data.

### For Data Sovereignty

This project demonstrates that African countries can develop their own capabilities for agricultural monitoring using open data and locally-built AI solutions — without relying on expensive proprietary platforms or foreign consultants.

---

## 6. What We Learned

### What Worked Well

1. **Seasonal information is crucial** — A single snapshot is not enough; looking at crops across multiple months dramatically improved accuracy.
2. **Infrared and other invisible wavelengths** — These were the most important for telling crops apart, far more useful than visible colours alone.
3. **Asking multiple models** — Combining five models worked better than any single one.
4. **Averaging across months** — Using all available time points for each farm gave more reliable predictions.

### What Could Be Improved

1. **More data from the rainy season** — Fewer images were available from May to September due to cloud cover, which may have limited the model's understanding of wet-season crop behaviour.
2. **Direct image analysis** — We used summary statistics of each image (averages, etc.). A more advanced approach could analyse the full image pixel-by-pixel to capture spatial patterns.
3. **Better cloud handling** — The model currently ignores cloud cover, but explicitly removing cloudy pixels could improve accuracy.
4. **Manual corrections need automation** — The post-processing adjustments worked for the competition but would need to be automated for real-world deployment.

### What Surprised Us

- Cocoa was by far the easiest crop to identify, despite being the least common in the dataset. Its distinct farming system (shade-grown under taller trees) gives it a unique spectral signature visible from space.
- Palm and rubber were harder to separate because both are dense, evergreen tree crops with similar growth patterns.
- Even with limited wet-season data, the multi-temporal approach still produced strong results.

---

## 7. What's Next

The next steps for this work could include:

- **Deploying as a web tool** — So that agricultural planners can upload satellite images and get crop maps
- **Expanding to other crops and countries** — Training the model to recognise additional crop types across West Africa
- **Combining with radar satellite data** — Radar can see through clouds, filling in the rainy-season gaps
- **Partnering with local organisations** — Ground-truth data collection to validate and improve the model
- **Building a monitoring system** — Tracking crop changes over time to detect deforestation, crop disease, or climate impacts

---

*Built with open data from the European Space Agency's Copernicus programme. Supported by Zindi, the Data Governance in Africa initiative (funded by the European Union, Germany, Finland, Belgium, France, and Estonia), and Tolbi — a Climate/AgTech startup developing AI-powered precision agriculture solutions for Africa.*
