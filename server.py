from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()

    if 'test' not in data:
        return jsonify({"error": "Missing 'test' parameter"}), 400

    test = data['test']

    try:
        zuw_ug, buruu_ug, ugnuud = ug_shalgah(test)
        stems = undes_ug(ugnuud)
        final_sentence = undes_ug_sentence(stems)
        suggestions = buruu_ug_sanal(buruu_ug)

        sedev = predict_news(test)

        response = {
            "zuw_ug": zuw_ug,
            "buruu_ug": buruu_ug,
            "final_sentence": final_sentence,
            "suggestions": suggestions,
            "sedev":sedev
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

app.run()