from flask import Flask, render_template, request, jsonify
import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load OpenAI API Key
my_secret = os.getenv('OPENAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json  # Get JSON data
        brand_name = data.get('brandName')
        product_description = data.get('description')
        target_audience = data.get('audience')

        # Ensure all fields are provided
        if not brand_name or not product_description or not target_audience:
            return jsonify({"error": "Missing required fields"}), 400

        # Define Prompt Template
        prompt_template = PromptTemplate(input_variables=[
            "brand_name", "product_description", "target_audience"
        ],
        template="""
            You are an expert marketing copywriter. Your job is to create a compelling ad copy.

            **Brand Name:** {brand_name}
            **Product/Service:** {product_description}
            **Target Audience:** {target_audience}

            ### **Output Format:**
            - **Ad Headline:** Catchy, engaging (under 10 words).
            - **Marketing Description:** Persuasive 2-3 sentences.
            - **Call-To-Action (CTA):** Strong CTA.

            Generate the best ad copy possible.
            """)

        # Initialize LLM
        llm = ChatOpenAI(model_name="gpt-4-turbo",
                         temperature=0.8,
                         max_tokens=150)
        marketing_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Run LLMChain
        response = marketing_chain.run({
            "brand_name": brand_name,
            "product_description": product_description,
            "target_audience": target_audience
        })

        return jsonify({"generated_text": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
