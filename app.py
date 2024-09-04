import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up Google API Key
GOOGLE_API_KEY = "AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU"

# Initialize the Gemini 1.5 Pro model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GOOGLE_API_KEY,
    temperature=0.8,  # Adjust temperature for more creative responses
    max_tokens=512,  # Adjust max tokens as needed
    timeout=None,
    max_retries=2,
)

# Define the game’s story templates and logic
story_template = """
You are the main character in an epic journey. Your adventure begins in a {setting}, where you encounter {conflict}. 
Your goal is to {goal}, but along the way, you face challenges that test your {skills}. 
The choices you make will determine the outcome of your story.

Here are the starting details:
Setting: {setting}
Conflict: {conflict}
Goal: {goal}

What would you like to do first?
"""

puzzle_template = """
A puzzle presents itself in the form of {puzzle_type}. Solve this to move forward in your journey.
The puzzle is: {puzzle_description}

Hint: Think about the clues you've encountered so far.
"""

# Define the PromptTemplate for dynamically generating the story
story_prompt = PromptTemplate(
    input_variables=["setting", "conflict", "goal", "skills"],
    template=story_template
)

# Define the PromptTemplate for generating dynamic puzzles
puzzle_prompt = PromptTemplate(
    input_variables=["puzzle_type", "puzzle_description"],
    template=puzzle_template
)

# Create LLMChains for the story and puzzles
story_chain = LLMChain(
    llm=llm,
    prompt=story_prompt
)

puzzle_chain = LLMChain(
    llm=llm,
    prompt=puzzle_prompt
)

# Streamlit interface for the game
st.title("Interactive Fiction Game with Dynamic Storylines")

st.write("### Welcome to your personalized adventure! 🎮")
st.write("Your journey will adapt to the choices you make, and you may encounter puzzles along the way.")

# Input fields for the player to create their own adventure elements
setting = st.text_input("Enter the setting for your story (e.g., a mystical forest, a futuristic city):", "a mystical forest")
conflict = st.text_input("Enter the conflict you face (e.g., a dragon terrorizing the land, an evil corporation plotting destruction):", "a dragon terrorizing the land")
goal = st.text_input("Enter your main goal (e.g., retrieve the ancient artifact, save the world):", "retrieve the ancient artifact")
skills = st.text_input("Enter the skills your character possesses (e.g., swordsmanship, magic, detective skills):", "magic")

# Button to start the game
if st.button("Start Your Adventure"):
    with st.spinner("Weaving your story..."):
        # Generate the first part of the story
        story_output = story_chain.run({
            "setting": setting,
            "conflict": conflict,
            "goal": goal,
            "skills": skills
        })
        
        st.write("### Your Story Begins:")
        st.write(story_output)
        
        # Simulate the appearance of a puzzle
        puzzle_trigger = st.checkbox("Encounter a Puzzle")
        
        if puzzle_trigger:
            puzzle_type = st.selectbox("Choose a type of puzzle:", ["riddle", "logic puzzle", "pattern recognition challenge"])
            puzzle_description = st.text_input(f"Describe the {puzzle_type} you want to encounter:", f"A riddle of ancient lore that must be solved to unlock the secret passage.")
            
            if st.button("Solve the Puzzle"):
                with st.spinner("Generating your puzzle..."):
                    # Generate the puzzle dynamically based on the user's choice
                    puzzle_output = puzzle_chain.run({
                        "puzzle_type": puzzle_type,
                        "puzzle_description": puzzle_description
                    })
                    st.write("### A Puzzle Appears:")
                    st.write(puzzle_output)
