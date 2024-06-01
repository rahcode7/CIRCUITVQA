symbol_position = [
    "Which circuit symbol is on the far XX?",
    "Identify the circuit symbol that is at the extreme XX.",
    "What is the circuit symbol located on the XXmost side?",
    "Tell me the circuit symbol positioned at the XXmost end.",
    "Point out the circuit symbol that is furthest to the XX.",
    "Which circuit symbol is on the very XX-hand side?",
    "Please indicate the circuit symbol situated all the way to the XX.",
    "What is the name of the circuit symbol at the XXmost position?",
    "Which circuit symbol is on the extreme XX?",
    "Find the circuit symbol that is farthest to the XX.",
    "Determine the circuit symbol on the XXmost side.",
    "Locate the circuit symbol positioned at the very XX.",
    "Can you tell me which circuit symbol is at the XXmost position?",
    "Which circuit symbol is placed at the extreme XX end?",
    "Point me to the circuit symbol on the XXmost side.",
    "What is the circuit symbol's name that appears on the XXmost?",
    "Show me the circuit symbol that is on the XXmost edge.",
    "Tell me the circuit symbol positioned to the far XX.",
    "Among the circuit symbols, which one is at the XXmost position?"
]


count_templates = [
'How many XXs are there in the specified circuit',
'What number of XX are included in the given circuit',
'What is the total count of XXs in the circuit?',
'Can you determine the number of XXs in the circuit?',
'How numerous are the XXs in the circuit?',
'What is the quantity of XXs present in the circuit?',
'Are there multiple XXs in the circuit?',
'What is the total XX count within the circuit?',
'Could you provide the number of XXs in the circuit?',
'How many components are there in the circuit that function as XXs?',
'What is the XX tally in the circuit?',
'Can you ascertain the number of XXs in the circuit?',
'Could you indicate the quantity of XXs present in the circuit?',
'How many XX devices are there in the circuit?',
'What is the total XX count in the given circuit?',
'Do you know how many XXs are present in the circuit?',
'Can you determine the number of XX components in the circuit?',
'Could you specify the quantity of XXs in the circuit?',
'Could you provide the count of XXs included in the circuit?'
#'What is the tally of components offering XX in the circuit?'
]

complex_count_templates  = {
    'd1' : ["How many YY are connected directly to the left of XX ?","How many YY are connected directly to the right of XX ?"],
    'd2' : ["How many YY are connected directly to the left of XX ?","How many YY are connected directly to the right of XX ?"],
    'd3' : ["How many YY are connected directly to the left of XX ?","How many YY are connected directly to the right of XX ?"],
    'd4' : ["How many gates are providing an input to XX", "How many gates are connected to the right of XX ?", "How many YY gates are connected to the right of XX ?", "How many YY gates are connected to the left of XX ?"],
    'd5' : ["How many YY are connected directly to the left of XX ?","How many YY are connected directly to the right of XX ?"]
}

## VALUE BASED 
value_templates = [
      "What are the current reading displayed by the XX?", 
"Please provide the values displayed on the XX.",
"What does the XX show in terms of reading?",
"What numerical value is being shown on the XX?",
"What reading does the XX display?",
"What are the value depicted on the XX?",
"Can you provide the current measurement given by the XX?",
"What are the current value indicated on the XX?",
"What does the XX read at the moment?",
"What are the present reading on the XX?",
"Could you share the current reading that the XX shows?"
]

## TOTAL VALUES
totalvalue_templates = ["What is the total XX in the circuit ?"]

junction_templates = ["Does a XX exist between junction YY and junction ZZ ?",
"Is there a XX present from junction YY to junction ZZ?"
"Does a XX occupy the space between junction YY and junction ZZ?"
"Is there a XX connecting junction YY to junction ZZ?"
"Can a XX be found between junction YY and junction ZZ?"
"Does junction YY have a XX leading to junction ZZ?"
"Is there a XX in the path from junction YY to junction ZZ?"
"Can we observe a XX between junction YY and junction ZZ?"
"Does the circuit between junction YY and junction ZZ contain a XX?"
"Is a XX situated between junction YY and junction ZZ?"
"Is there impedance in the connection between junction YY and junction ZZ?"
"Can you confirm the presence of a XX between junction YY and junction ZZ?"
"Is there any resistance between junction YY and junction ZZ?"
"Does the circuit at junction YY involve a XX leading to junction ZZ?"
"Is a XX located along the path from junction YY to junction ZZ?"
"Can you verify if there is a XX between junction YY and junction ZZ?"
"Is a XX part of the circuit between junction YY and junction ZZ?"
"Is there a XX linking junction YY to junction ZZ?"
"Is there a XX bridging the gap between junction YY and junction ZZ?"
"Does junction YY connect to junction ZZ through a XX?"
"Is there any resistance encountered from junction YY to junction ZZ?"
"Is a XX placed in the line connecting junction YY and junction ZZ?"
"Does a XX exist within the path from junction YY to junction ZZ?"
"Can you confirm the existence of a XX between junction YY and junction ZZ?"
"Is there a XX interposed between junction YY and junction ZZ?"
"Does junction YY have a XX leading to junction ZZ?"
"Is there a XX that separates junction YY from junction ZZ?"
"Is there impedance in the path between junction YY and junction ZZ?"
"Is a XX present in the pathway from junction YY to junction ZZ?"
"Does the circuit at junction YY involve a XX in connection with junction ZZ?"
"Is there any resistance present between junction YY and junction ZZ?"
]