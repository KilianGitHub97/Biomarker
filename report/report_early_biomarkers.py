from fpdf import FPDF

pdf = FPDF(orientation="P", unit="mm", format="A4")
class utils:
    
    @staticmethod
    def header():
        pdf.set_font("Times", "", 7)
        pdf.set_left_margin(15)
        pdf.cell(w = 60, h = 0, txt = "Kilian Sennrich", align="L")
        pdf.cell(w = 60, h = 0, txt = "Early Parkinson", align="C")
        pdf.cell(w = 60, h = 0, txt = "18.07.2021", align="R")

    
    
    
def report():
    #setup
    pdf.set_title("Early Biomarkers for Morbus Parkinson")
    pdf.set_author("Kilian Sennrich")
    pdf.set_keywords(keywords = "Biomarker, Parkinson, Machine Learning")
   
    #first page
    pdf.add_page()
    utils.header()
    
    pdf.set_font('Times', 'B', 16)
    pdf.write(h = 15, txt="Hello This is my fist report in Python.")
    
    #second page
    
    #out
    pdf.output('early_biomarkers.pdf', 'F')

report()
