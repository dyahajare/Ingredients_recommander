# backend_endpoint.py (Updated)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import traceback
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import IDFModel, StandardScalerModel, Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType
import pickle
import re

app = Flask(__name__)
CORS(app)

spark = None
recommender_system = None
tfidf_model_loaded = None
scaler_model_loaded = None
num_features_for_ingredients_loaded = None
df_products_vectorized_loaded = None

def initialize_spark_and_models():
    global spark, recommender_system, tfidf_model_loaded, scaler_model_loaded, num_features_for_ingredients_loaded, df_products_vectorized_loaded

    if spark is not None:
        print("Spark session already initialized.")
        return

    print("Initializing Spark session and loading models...")

    try:
        spark = SparkSession.builder \
            .appName("Open Food Facts Recommendation System Backend") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.instances", "1") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.rpc.message.maxSize", "512") \
            .config("spark.network.timeout", "300s") \
            .config("spark.driver.host", "127.0.0.1") \
            .getOrCreate()

        gdrive_path = "../recommender_system/"

        with open(os.path.join(gdrive_path, "metadata.pickle"), 'rb') as f:
            metadata = pickle.load(f)
        num_features_for_ingredients_loaded = metadata['num_features_for_ingredients']

        tfidf_model_loaded = IDFModel.load(os.path.join(gdrive_path, "tfidf_model"))
        scaler_model_loaded = StandardScalerModel.load(os.path.join(gdrive_path, "scaler_model"))
        df_products_vectorized_loaded = spark.read.parquet(os.path.join(gdrive_path, "df_products_vectorized"))
        df_products_vectorized_loaded.cache()
        df_products_vectorized_loaded.count()

        def get_recommendations_backend(user_recipe, country=None, nutriscore_filter=None, allergen_exclusions=None, top_k=10):
            print(f"Generating recommendations for: '{user_recipe}' in country: {country}")

            user_recipe_clean_str = re.sub(r'[^a-zA-Z0-9]', '', user_recipe).lower()
            user_df_spark = spark.createDataFrame([(user_recipe_clean_str,)], ["ingredients_clean"])

            tokenizer = Tokenizer(inputCol="ingredients_clean", outputCol="words")
            user_words_data = tokenizer.transform(user_df_spark)

            remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
            user_filtered_words = remover.transform(user_words_data)

            hashingTF_user = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=num_features_for_ingredients_loaded)
            user_raw_features = hashingTF_user.transform(user_filtered_words)

            user_ingredient_vector_df = tfidf_model_loaded.transform(user_raw_features)
            user_ingredient_vector_list = user_ingredient_vector_df.limit(1).collect()

            user_ingredient_vector = user_ingredient_vector_list[0].ingredient_vectors if user_ingredient_vector_list else Vectors.sparse(num_features_for_ingredients_loaded, [], [])

            nutrition_features_length = len(scaler_model_loaded.mean)
            user_nutrition_vector = Vectors.dense([0.0] * nutrition_features_length)

            broadcast_ingredient_vector = spark.sparkContext.broadcast(user_ingredient_vector)
            broadcast_nutrition_vector = spark.sparkContext.broadcast(user_nutrition_vector)

            @F.udf(DoubleType())
            def dynamic_ingredient_similarity_udf(product_vector):
                user_vec_b = broadcast_ingredient_vector.value
                if product_vector is None or user_vec_b is None or product_vector.size != user_vec_b.size:
                    return 0.0
                dot_product = float(product_vector.dot(user_vec_b))
                norm_product = float(product_vector.norm(2)) * float(user_vec_b.norm(2))
                return dot_product / norm_product if norm_product != 0 else 0.0

            @F.udf(DoubleType())
            def dynamic_nutrition_similarity_udf(product_vector):
                user_vec_b = broadcast_nutrition_vector.value
                if product_vector is None or user_vec_b is None or product_vector.size != user_vec_b.size:
                    return 0.0
                dot_product = float(product_vector.dot(user_vec_b))
                norm_product = float(product_vector.norm(2)) * float(user_vec_b.norm(2))
                return dot_product / norm_product if norm_product != 0 else 0.0

            df_ranked = df_products_vectorized_loaded.withColumn(
                "ingredient_similarity", dynamic_ingredient_similarity_udf(F.col("ingredient_vectors"))
            ).withColumn(
                "nutrition_similarity", dynamic_nutrition_similarity_udf(F.col("nutrition_vectors"))
            ).withColumn(
                "combined_similarity_score",
                (F.lit(0.6) * F.col("nutrition_similarity")) + (F.lit(0.4) * F.col("ingredient_similarity"))
            )

            if country:
                df_ranked = df_ranked.filter(F.lower(F.col('countries')).contains(country.lower()))

            if nutriscore_filter:
                if isinstance(nutriscore_filter, str):
                    nutriscore_filter = [nutriscore_filter]
                nutriscore_filter_lower = [g.lower() for g in nutriscore_filter]
                df_ranked = df_ranked.filter(F.lower(F.col('nutriscore_grade')).isin(nutriscore_filter_lower))

            final_recommendations_spark = df_ranked.orderBy(F.col("combined_similarity_score").desc()).limit(top_k)

            result_df = final_recommendations_spark.select(
                'product_name', 'categories', 'nutriscore_grade',
                'combined_similarity_score', 'countries', 'brands', 'image_url'
            ).toPandas()

            broadcast_ingredient_vector.unpersist()
            broadcast_nutrition_vector.unpersist()

            return result_df

        recommender_system = get_recommendations_backend
        print("Recommender system function created.")

    except Exception as e:
        print(f"Error during Spark initialization or model loading: {e}")
        traceback.print_exc()
        spark = None

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"}), 200

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        initialize_spark_and_models()
        if recommender_system is None:
            return jsonify({"error": "Recommender system not initialized."}), 500

        data = request.get_json()
        user_recipe = data.get("recipe")
        country = data.get("country")
        nutriscore_filter = data.get("nutriscore_filter")
        allergen_exclusions = data.get("allergen_exclusions")

        result_df = recommender_system(user_recipe, country, nutriscore_filter, allergen_exclusions)
        result = result_df.to_dict(orient="records")
        return jsonify(result), 200

    except Exception as e:
        print("Error in /api/recommend endpoint:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
