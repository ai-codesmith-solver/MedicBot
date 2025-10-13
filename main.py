from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)


@app.route("/") 
def main():
    return render_template("main.html")

chat_history=[]
@app.route("/chat", methods=["POST"])
def chat():
    user_inp = request.form.get("text")
    chat_history.append({'user':user_inp})
    def ai(user_inp):
        a4f_api_key = "Enter-Your-API-Key"
        a4f_base_url = "https://api.a4f.co/v1"

        client = OpenAI(
            api_key=a4f_api_key,
            base_url=a4f_base_url,
        )

        completion = client.chat.completions.create(
            model="provider-1/gemma-3-4b-it",
            messages=[
                {
                    "role": "user",
                    "content": f"{user_inp}\n\nYou are MedicBot 🩺—a professional and trusted virtual doctor. Respond briefly and accurately in one line, using medically sound advice in simple language. Be clear, calm, and reassuring like a real doctor. Only include emojis if truly helpful. The Chat History:\n{chat_history}",
                }
            ]
        )
        return completion.choices[0].message.content
    
    bot_res=ai(user_inp)
    chat_history.append({'bot':bot_res})
    return bot_res


if __name__ == "__main__":
    app.run(debug=True)
