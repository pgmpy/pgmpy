"""
ProbModelXML: http://leo.ugr.es/pgm2012/submissions/pgm2012_submission_43.pdf

For the student example the ProbModelXML file should be:

<?xml version=“1.0” encoding=“UTF-8”?>
<ProbModelXML formatVersion=“1.0”>
    <ProbNet type=BayesianNetwork >
        <AdditionalConstraints />
        <Comment>
            Student example model from Probabilistic Graphical Models: Principles and Techniques by Daphne Koller
        </Comment>
        <Language>
            English
        </Language>
        <AdditionalProperties />
        <Variable name="intelligence" type="FiniteState" role="Chance">
            <Comment />
            <Coordinates />
            <AdditionalProperties />
            <States>
                <State name="smart"><AdditionalProperties /></State>
                <State name="dumb"><AdditionalProperties /></State>
            </States>
        </Variable>
        <Variable name="difficulty" type="FiniteState" role="Chance">
            <Comment />
            <Coordinates />
            <AdditionalProperties />
            <States>
                <State name="difficult"><AdditionalProperties /></State>
                <State name="easy"><AdditionalProperties /></State>
            </States>
        </Variable>
        <Variable name="grade" type="FiniteState" role="Chance">
            <Comment />
            <Coordinates />
            <AdditionalProperties />
            <States>
                <State name="grade_A"><AdditionalProperties /></State>
                <State name="grade_B"><AdditionalProperties /></State>
                <State name="grade_C"><AdditionalProperties /></State>
            </States>
        </Variable>
        <Variable name="recommendation_letter" type="FiniteState" role="Chance">
            <Comment />
            <Coordinates />
            <AdditionalProperties />
            <States>
                <State name="good"><AdditionalProperties /></State>
                <State name="bad"><AdditionalProperties /></State>
            </States>
        </Variable>
        <Variable name="SAT" type="FiniteState" role="Chance">
            <Comment />
            <Coordinates />
            <AdditionalProperties />
            <States>
                <State name="high"><AdditionalProperties /></State>
                <State name="low"><AdditionalProperties /></State>
            </States>
        </Variable>
        <Links>
            <Link var1="difficulty" var2="grade" directed=1>
                <Comment>Directed Edge from difficulty to grade</Comment>
                <Label>diff_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="grade" directed=1>
                <Comment>Directed Edge from intelligence to grade</Comment>
                <Label>intel_to_grad</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="intelligence" var2="SAT" directed=1>
                <Comment>Directed Edge from intelligence to SAT</Comment>
                <Label>intel_to_sat</Label>
                <AdditionalProperties />
            </Link>
            <Link var1="grade" var2="recommendation_letter" directed=1>
                <Comment>Directed Edge from grade to recommendation_letter</Comment>
                <Label>grad_to_reco</Label>
                <AdditionalProperties />
            </Link>
        </Links>
        <Potential type="Table" role="ConditionalProbability" label=string>
            <Comment>CPDs in the form of table</Comment>
            <AdditionalProperties />
            <!--
                There is no specification in the paper about how the tables should be represented.
            -->
        </Potential>
    </ProbNet>
    <Policies />
    <InferenceOptions />
    <Evidence>
        <EvidenceCase>
            <Finding variable=string state=string stateIndex=integer numericValue=number/>
        </EvidenceCase>
    </Evidence>
</ProbModelXML>
"""