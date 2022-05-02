import streamlit as st
import transformers

model_name = 'facebook/wmt19-ru-en'

@st.cache
def load_pipeline():
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    def translate(input_string):
        encode_obj = tokenizer(input_string, return_tensors='pt').input_ids
        outputs = model.generate(encode_obj)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translate


if __name__ == '__main__':
    pipeline = load_pipeline()
    st.title('Multilingual Translation')

    user_input = st.text_input("Enter the sentence you would like to translate.", "translate from English to Russian: ")
    if st.button("Translate!"):
        st.write(pipeline(user_input))


