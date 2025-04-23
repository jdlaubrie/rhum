from IPython.display import clear_output
import os

def installpkgs():
    try:
        import dolfin
    except ImportError:
        print("Installing fenics... this should take about 30-60 seconds.")
        os.system('wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')

    try:
        import gmsh
    except ImportError:
        print("Installing gmsh... this should take about 30-60 seconds.")
        os.system('wget "https://fem-on-colab.github.io/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"')

    clear_output(wait = True)
    print("Both fenics and gmsh are installed.")
