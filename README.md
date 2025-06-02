# Ingredients_recommander
ChefAI: An intelligent ingredient and brand recommendation system. Leverages Ollama Llama 3.2 for local recipe extraction and Apache Spark for advanced data analysis (TF-IDF) on Open Food Facts, providing precise, context-aware suggestions.
Ah, that clarifies the technical stack and process significantly! Integrating a local LLM (Ollama Llama 3.2) and specifying Spark's role beyond general data processing adds crucial detail.

Le Cœur de l'Innovation :

Extraction d'Ingrédients par LLM Local (Ollama Llama 3.2) :

Au premier plan, ChefAI intègre le modèle de langage Ollama Llama 3.2 exécuté localement. Cela permet une analyse rapide et privée de vos recettes.
Lorsque vous soumettez une recette (copier-coller un texte, par exemple), Llama 3.2 extrait intelligemment et avec précision les ingrédients, leurs quantités et les unités de mesure, même à partir de textes complexes ou moins structurés. Cette capacité d'extraction avancée est cruciale pour la fiabilité de tout le système.
Moteur de Recommandation d'Ingrédients et de Marques (Open Food Facts & Spark) :

Les ingrédients extraits sont ensuite le point de départ d'un puissant moteur de recommandation.
Intégration avec Open Food Facts : ChefAI se connecte à la vaste base de données d'Open Food Facts pour récupérer des informations détaillées sur chaque ingrédient. Cela inclut des données nutritionnelles, la présence d'allergènes, les labels de qualité (bio, végétarien, équitable), l'origine, et crucialement, les noms de marques disponibles sur le marché.
Optimisation par Apache Spark : L'intelligence derrière les recommandations de marques et la compréhension des relations entre les ingrédients réside dans Apache Spark. Spark est utilisé pour :
Exploration et Préparation des Données : Nettoyage et structuration des vastes jeux de données d'Open Food Facts et des bases de données de recettes.
Traitement du Langage Naturel (NLP) Avancé : Application de techniques comme TF-IDF (Term Frequency-Inverse Document Frequency) pour comprendre la pertinence des termes et des associations d'ingrédients. Cela permet d'identifier non seulement les ingrédients requis mais aussi de suggérer des marques pertinentes en fonction du contexte de la recette ou de préférences implicites.
Détection de Tendances et de Relations : Spark permet d'analyser les co-occurrences d'ingrédients et de marques dans des milliers de recettes, offrant des recommandations plus intelligentes et contextuelles (par exemple, quelle marque de farine est souvent utilisée avec quelle marque de levure pour un certain type de pain).
Avantages Distinctifs de ChefAI :

Précision et Pertinence : Grâce à Llama 3.2 et Spark, les recommandations sont hautement précises et adaptées à votre recette.
Recommandations de Marques Concrètes : Passez de la simple liste d'ingrédients à des suggestions de marques disponibles, simplifiant ainsi vos achats.
Informations Complètes : Accédez instantanément à des détails nutritionnels, allergènes et certifications, vous aidant à faire des choix sains et responsables.
Vie Privée Respectée : L'utilisation d'un LLM local avec Ollama signifie que vos données de recettes ne quittent pas votre environnement, garantissant une confidentialité accrue.
Évolutivité : L'architecture basée sur Spark assure que l'application peut gérer et analyser des quantités de données croissantes sans compromettre les performances.
Pour Qui ?

ChefAI est l'outil idéal pour les cuisiniers amateurs et expérimentés, les personnes soucieuses de leur alimentation et de l'origine de leurs produits, et tous ceux qui souhaitent rationaliser leur processus de planification de repas et de courses avec l'aide d'une technologie de pointe.

ChefAI n'est pas seulement une application, c'est votre partenaire culinaire intelligent, vous guidant de l'inspiration de la recette à l'assiette finale, avec une connaissance approfondie des ingrédients et des produits qui rendent chaque plat parfait.

Merci de telecharger le fichier df_products_vectorized\part-00000-9b1a6cc0-c69a-44f9-86f7-51167c19421f-c000.snappy.parquet from the link below https://drive.google.com/file/d/1FyF_1tpUinMZgBOkinWDkDF-sOqejpkl/view?usp=sharing



