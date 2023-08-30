import os
import git

def getf (link):
    import os
    import git
    cwd =os.getcwd()
    url = "https://github.com/Microsoft/calculator.git"

    baseDoxyFile = cwd + "/Doxyfile"
    dir_name = url.split("/")[-1].split(".")[0]
    print(dir_name)
    #git.Git(cwd + '/softwares').clone(url)
    os.chdir(os.getcwd()+"/softwares")
    print(os.getcwd())
    os.system("git clone " +url)
    os.chdir(os.getcwd() + "/" + dir_name)
    print (os.getcwd())
    gendoxy = 'cp ' + baseDoxyFile + os.getcwd()
    os.system(gendoxy)
    cmd = 'doxygen Doxyfile'
    os.system(cmd)
    os.chdir(os.getcwd() + "/xml")
    os.system('ls')
