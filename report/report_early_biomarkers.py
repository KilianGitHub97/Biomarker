from fpdf import FPDF

pdf = FPDF(orientation="P", unit="mm", format="A4")

PAGES = 1
AUTHOR = "Kilian Sennrich"
TITLE = "Early Biomarkers for Morbus Parkinson"
TITLE2 = "Analysis of the data from Jan Hlavnicka et al. (2017)"
KEYWORDS = "Biomarker, Parkinson, Machine Learning"
DATE = "18.07.2021"

#Layout options
FONT = "Times"

class utils:
    
    @staticmethod
    def header():
        pdf.set_font(FONT, "", 7)
        pdf.set_left_margin(15)
        pdf.cell(w = 60, h = 0, txt = AUTHOR, align="L")
        pdf.cell(w = 60, h = 0, txt = TITLE, align="C")
        pdf.cell(w = 60, h = 0, txt = DATE, align="R")
        
    @staticmethod
    def footer():
        global PAGES
        pdf.set_font(FONT, "", 7)
        pdf.set_right_margin(-15)
        pdf.set_y(-10)
        pdf.cell(w = 180, h = 0, txt = "page: {}".format(PAGES), align="R")
        PAGES += 1
        
    @staticmethod
    def h1(txt): #general title
        pdf.set_font(family = FONT, style = "B", size=16)
        pdf.cell(h = 0, w = 0, txt = txt)
    
    @staticmethod
    def h2(txt): #section title
        pdf.set_font(family = FONT, style = "B", size = 12)
        pdf.cell(h = 0, w = 0, txt = txt)
        
    @staticmethod
    def h3(txt): #subsection title 1
        pdf.set_font(family = FONT, style = "", size = 12)
        pdf.cell(h = 0, w = 0, txt = txt)
    
    @staticmethod
    def h4(txt): #subsection title 2
        pdf.set_font(family = FONT, style = "I", size = 11)
        pdf.cell(h = 0, w = 0, txt = txt)
        
    @staticmethod
    def h5(txt): #paragraph title
        pdf.set_font(family = FONT, style = "B", size = 11)
        pdf.cell(h = 0, w = 0, txt = txt)
        
    @staticmethod
    def text(txt):
        pdf.set_font(family = FONT, style = "", size = 11)
        pdf.set_margins(15, 0, 15)
        pdf.write(h = 5, txt = txt)
    
    
def report():
    #setup
    pdf.set_title(TITLE)
    pdf.set_author(AUTHOR)
    pdf.set_keywords(keywords = KEYWORDS)
    pdf.set_auto_page_break(auto = True, margin = 0.0)

    #first page
    pdf.add_page()
    utils.header()    
    pdf.ln(20)
    utils.h1(txt = TITLE)
    pdf.ln(5)
    utils.h3(txt = TITLE2)
    pdf.ln(5)
    utils.h4(txt = AUTHOR + " / " + DATE)
    pdf.ln(10)
    utils.text(txt = 
               "This is the report for the project \"Early Biomarkers for Morbus Parkinson\" that I " +
               " have conducted to introduce myself to the python data-science libraries. " +
               "The data stems from a research paper on automatic speech processing from Hlavnicka et al. (2017) " +
               "The data was collected between 2014 ans 2016 in the Czech Republic. The study included " +
               "30 recently diagnosed and untreated Morbus Parkinson patients, idiopathic RBD " +
               "(which is a strong predictor of Morbus Parkinson) and 50 healthy controls. All " +
               "probands participated in two speeking tasks (reading, monologue). The goal of the study " +
               "was to automate the analysis of recordings of the patients, which worked out very well. " +
               "Hlavnicka et al. (2017) published the dataset within their appendix and it was reposted on " +
               "Kaggle. The research paper and the kaggle dataset can be accessed via the following links:"
               "\n \n"
               "-https://www.nature.com/articles/s41598-017-00047-5 \n"
               "-https://www.kaggle.com/ruslankl/early-biomarkers-of-parkinsons-disease")
    pdf.ln(15)
    utils.h2(txt="Table of Contents")
    utils.text(txt=
               """1.)  Introduction \n2.)  Descriptive Statistics \n3.)  Mean analysis \n4.)  Classification \n5.)  Recoverability of labels""")
    pdf.ln(15)
    utils.h2(txt="Introduction")
    utils.text(txt = 
               "From Kaggle: The dataset includes 30 patients with early untreated Parkinson's " +
               "disease (PD), 50 patients with REM sleep behavior disorder (RBD), " +
               "which are at high risk developing Parkinson's disease or other " +
               "synucleinopathies; and 50 healthy controls (HC). " +
               "All patients were scored clinically by a well-trained professional " +
               "neurologist with experience in movement disorders. All subjects " +
               "were examined during a single session with a speech specialist. " + 
               "All subjects performed reading of standardized, phonetically- " +
               "balanced text of 80 words and monologue about their interests, " +
               "job, family or current activities for approximately 90 seconds. \n" +
               "Inspiration: Predict a pattern of neurodegeneration in the dataset of speech " +
               "features obtained from patients with early untreated Parkinson's " +
               "disease and patients at high risk developing Parkinson's disease.")
    pdf.ln(10)
    utils.h4(txt = "Hoehn-and-Yahr Scale - A Measure for Morbus Parkinson")
    utils.text(txt=
               "The state of the Morbus Parkinson disease can be examined with different clinical assessments:" +
               "The Hoehn-and-Yahr scale is used to classify the severity of PD based on symptoms. It is an " +
               "easy-to-perform clinical instrument to assess the underlying movement disorders. " +
               "The Hoehn and Yahr scale was developed in 1967. It divides the " +
               "disease into 5 stages. So-called modified stages according to Hoehn " +
               "and Yahr (1987) are also frequently used (https://flexikon.doccheck.com/de/Hoehn-und-Yahr-Skala). The 5 states are: \n\n" +
               "   1)      Unilateral symptomatology \n" +
               "   1.5)    Unilateral symptoms and axial involvement \n" +
               "   2)      Bilateral symptoms; no postural instability \n" +
               "   2.5)    Mild bilateral symptoms; patient can regain balance on pull test (compensates on pull test) \n" +
               "   3)      Mild to moderate bilateral symptomatology; mild postural instability; independence maintained \n" +
               "   4)      Severe disability, but patient can still walk and stand without assistance \n" +
               "   5)      Patient is wheelchair bound or bedridden without assistance from others")
    


    utils.footer()
    
    #second page
    pdf.add_page()
    pdf.ln(10)
    utils.header()
    pdf.ln(10)
    utils.h4("UPDRS III total - A More General Measure")
    utils.text(txt=
               "The UPDRS III total scale is a more general scale to measure the state of the " +
               "Morbus Parkinson disearse. The Hoehn-and-Yahr Scale is integrated into the UPDRS III total, " +
               "as a sub-examination. The UPDRS scale refers to Unified Parkinson Disease Rating Scale, " +
               "and it is a rating tool used to gauge the course of Parkinson's disease in patients. " +
               "The UPDRS scale has been modified over the years by several medical organizations, " +
               "and continues to be one of the bases of treatment and research in PD clinics. The UPDRS scale " +
               "includes series of ratings for typical Parkinson's symptoms that cover all of the movement hindrances " +
               "of Parkinson's disease. The UPDRS scale consists of the following five segments: \n" +
               "1) Mentation, Behavior, and Mood, 2) ADL, 3) Motor sections, 4) Modified Hoehn and Yahr Scale, " +
               "and 5) Schwab and England ADL scale. \n" +
               "Each answer to the scale is evaluated by a medical professional that specializes in Parkinson's disease " + 
               "during patient interviews. Some sections of the UPDRS scale require multiple grades assigned " +
               "to each extremity with a possible maximum of 199 points. A score of 199 on the UPDRS scale represents " +
               "the worst (total disability) with a score of zero representing (no disability). \n"
               "(Copied from: https://www.theracycle.com/resources/links-and-additional-resources/updrs-scale/)")
    utils.footer()
    #out
    pdf.output('early_biomarkers.pdf', 'F')

report()
