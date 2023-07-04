from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.callbacks import CometCallbackHandler

# These lines import various classes from the LangChain library. The classes imported are ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ConversationChain, ChatOpenAI, and ConversationBufferMemory. These classes are used to define chat prompts, handle conversation chains, and manage conversation memory.
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.schema import AIMessage, HumanMessage, SystemMessage
import configparser




# Create a configparser instance
config = configparser.ConfigParser()
config.read('../config.ini')
api_key_input = config.get('default', 'lchain_api_key')
prompt_menu = config.get('prompt_texts', 'prompt_menu')
prompt_user_complaint = config.get('prompt_texts', 'prompt_user_complaint')
prompt_user_complaint_tuned_json = config.get('prompt_texts', 'prompt_user_complaint_tuned_json')
classification_prompt = config.get('prompt_texts', 'classification_prompt')
classification_prompt_enhanced = config.get('prompt_texts', 'classification_prompt_enhanced')
responding_to_customer = config.get('prompt_texts', 'responding_to_customer')

#comet_callback = CometCallbackHandler(
#    project_name="comet-example-langchain",
#    stream_logs=True,
##    complexity_metrics=True,
#    tags=["llm"],
#    visualizations=["dep"],
#)


def setup_chat_prompt(promp_input):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(promp_input),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    return prompt

def menu_order():
    global api_key_input
    global prompt_menu
    global comet_callback
    sys_prompt = setup_chat_prompt(prompt_menu)
    prompt_array = [
        "Do you have a kids' menu?",
        "Do you have any vegan options?",
        "How much for the shoes?",
        "Do you have mac & cheese?",
        "What's the price for the BBQ?",
        "What's the price for the mac & cheese?",
        "What's your most popular dish?"
    ]

    llm = OpenAI(temperature=0.9, callbacks=[comet_callback], openai_api_key=str(api_key_input))
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=sys_prompt, llm=llm)

    for convo in prompt_array:
        print(f"Question: {convo}")
        response = conversation.predict(input=convo)
        print(response)
        print("-----------------------------------")

def customer_complaint():
    global api_key_input
    global prompt_user_complaint
    global comet_callback

    sys_prompt = setup_chat_prompt(prompt_user_complaint)

    complaint_text = "### I ordered a pair of shoes two weeks ago and still haven't received them. The tracking information hasn't been updated in days and I have no idea where my package is. ###"
    complaint_text_2 = "### I love my job ###"

    #llm = OpenAI(temperature=0.9,  callbacks=[comet_callback], openai_api_key=str(api_key_input))
    llm = OpenAI(temperature=0.9, openai_api_key=str(api_key_input))
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=sys_prompt, llm=llm)

    print(complaint_text)
    response = conversation.predict(input=complaint_text)
    print(response)

def classify_complaints():
    global api_key_input
    global classification_prompt #1 
    global classification_prompt_enhanced #2
    #global comet_callback

    sys_prompt = setup_chat_prompt(classification_prompt_enhanced)

    complaint_list_user_prompt= ["I ordered a pair of shoes two weeks ago and still haven't received them. The tracking information hasn't been updated in days and I have no idea where my package is.",
                                 "I lost my money. I cannot afford to pay my rent.", "These socks absolutely suck!!", "I ordered green and this blazer look red", "It says £10.99 for the t-shirt but my bank statement says £11.99"]

    #llm = OpenAI(temperature=0.9, callbacks=[comet_callback], openai_api_key=str(api_key_input))
    llm = OpenAI(temperature=0.9, openai_api_key=str(api_key_input))
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=sys_prompt, llm=llm)

    for complaint in complaint_list_user_prompt:
        response = conversation.predict(input=complaint)
        print(response)

def respond_to_user():
    global api_key_input
    global responding_to_customer

    user_complaint = [{
                        "ticket_no": "123456",
                        "customer_name": "John Doe",
                        "user_complaint": "I lost my money. I cannot afford to pay my rent.",
                        "priority": "3",
                        "category": "N/A",
                        "product_mentions": "N/A",
                        "comments": "I am not sure if this is really a complaint of our service"
                    }, 
                    {
                        "ticket_no": "102839",
                        "customer_name": "Amy Winehouse",
                        "user_complaint": "I ordered a pair of shoes two weeks ago and still haven't received them. The tracking information hasn't been updated in days and I have no idea where my package is.",
                        "priority": "3",
                        "category": "Late or Non-Delivery",
                        "product_mentions": "shoes",
                        "comments": "shoes may have been lost in transit, apologize to the customer and offer a refund or replacement"
                    
                    }
                    ]

    sys_prompt = setup_chat_prompt(responding_to_customer)
    llm = OpenAI(temperature=0.9, openai_api_key=str(api_key_input))
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=sys_prompt, llm=llm)
    print("------ Response------")
    for complaint in user_complaint:
        complaint = str(complaint)
        
        response = conversation.predict(input=complaint)
        print(response)
        print("---------------------")


def main():
    #menu_order()
    #customer_complaint()
    #classify_complaints()
    respond_to_user()

if __name__ == '__main__':
    main()
