from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

# selected_model="llama3:8b"

selected_model="mistral:7b"



the_prompt = """ You are an expert text prompt generator. I need you to generate text for prompts based on the following context. Below are first the template prompt followed by the variables for the template. The variables X, Y, Z, A, B, I1, I2 are defined after the template. The template is in quotes. To generate the text for the prompt please pick values ar random for the variables based on the types for the variables stated below. Do not create the clinical note. Generate a prompt with populated variables that will be used to create a clinical note later.

“You are a clinician specializing in epilepsy patients. You will generate a clinical note for a patient of X, Y, Z characteristics with Symptoms and Co-morbidities A, B. 

The patient should be such that we should be able to decide on whether to give this patient Cenbomate based on the following sources of information: I1, I2

The patient note must include the patient’s first and last names and a patient ID.”


Note again: Do not create the clinical note. Generate a prompt with populated variables that will be used to create a clinical note later.

Also do not recommend a medication. Limit it to notes about the patient. Include the sources of information in the prompt.

You will pick values at random for all the variables based on the following.

Variable X is age
Variable Y is sex
Variable Z is race

The symptoms A and co-morbidities B.are in the following table. Pick a pair where the numerical index of the symptom and comorbid condition match up.
 
Symptoms	
    1.   Rash
	2.	Photosensitivity
	3.	Sleep Disturbances
	4.	Fatigue
	5.	Memory Loss
	6.	Muscle Weakness
	7.	Headaches
	8.	Mood Swings
	9.	Dizziness
	10.	Confusion
	11.	Tremor
	12.	Difficulty Speaking
	13.	Nausea
	14.	Aggression
	15.	Visual Disturbances
	16.	Palpitations
	17.	Sweating
	18.	Weight Loss
	19.	Weight gain
	20.	Hearing Loss
	21.	Back Pain
	22.	Chest Pain
	23.	Joint Pain
	24.	Auditory Hallucinations
	25.	Visual Hallucinations	

Comorbid conditions
    1.	Eczema
	2.	Lupus
	3.	Insomnia
	4.	Chronic Fatigue Syndrome
	5.	Alzheimer’s Disease
	6.	Multiple Sclerosis
	7.	Migraines
	8.	Bipolar Disorder
	9.	Benign paroxysmal positional vertigo
	10.	Reactive Hypoglycemia
	11.	Essential Tremor
	12.	Aphasia
	13.	Gastrointestinal reflux disease
	14.	Intermittent Explosive Disorder
	15.	Glaucoma
	16.	Arrhythmia
	17.	Hyperhidrosis
	18.	Hyperthyroidism
	19.	Hypothyroidism
	20.	Meniere’s Disease
	21.	Herniated Disc
	22.	Angina
	23.	Rheumatoid Arthritis
	24.	Schizophrenia
	25.	Drug Abuse



I1: colleague experience data: “Cenobamate should be used only as a second line agent in focal epilepsy.”
“Cenobamate has significant drug-drug interactions with enzyme inducing anti-seizure medications, therefore it might not be considered second line if the first medication was an enzyme inducer.”
“When a patient has failed multiple anti-seizure medications, cenobamate may be a better choice than other anti-seizure medications.”
I2: literature data: 
Information	cenobamate max dose	Percentage Difference from Placebo
median seizure reduction more than placebo	200 mg	34.1%
somnolence	200 mg	10.2%
dizziness	200 mg	5.6%
nausea	200 mg	6.9%
fatigue	200 mg	4.2%
nystagmus	200 mg	9.7%
balance disorder	200 mg	7.1%
urinary tract infection	200 mg	6.2%
nasopharyngitis	200 mg	5.3%
tremor	200 mg	3.4%
constipation	200 mg	5.3%
diarrhea	200 mg	5.3%
vomiting	200 mg	3.5%

"""

def main():

    llm = Ollama(
    model=selected_model, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=1
    )

    print("getting prompt together")
    gen_prompt = llm.invoke(the_prompt)

    print("Generated prompt::", gen_prompt)

if __name__ == "__main__":
    main()