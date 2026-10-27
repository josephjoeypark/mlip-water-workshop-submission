"""Combine rebuilt vector figures and the archived explanatory notes."""
import shutil
from pypdf import PdfWriter
from config import OUT, FIG, FROZEN

NAMES = ['figure1_structure','figure2_transport','figureA1_molecular','figureS1_raw_transport',
         'figureS2_stress_short','figureS3_stress_tail','figureS4_green_kubo_50ps','figureS5_green_kubo_20ps',
         'figureS6_vacf','figureS7_running_diffusion','figureS8_disjoint_viscosity','figureS9_disjoint_diffusion',
         'figureS10_block_length','figureS11_prefix_length_295K']

def main():
    for filename, names, notes in [('paper_figures.pdf', NAMES[:3], False), ('final_2ns_report.pdf', NAMES, True)]:
        writer = PdfWriter()
        if notes:
            writer.append(FROZEN / 'report_notes.pdf')
        for name in names:
            writer.append(FIG / (name + '.pdf'))
        writer.write(OUT / filename)
        writer.close()
    shutil.copyfile(FROZEN / 'FIGURE_CAPTIONS.txt', OUT / 'FIGURE_CAPTIONS.txt')

if __name__ == '__main__':
    main()
