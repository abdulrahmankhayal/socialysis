import os
import shutil

cwd = os.path.dirname(__file__)


def get_sw():
    """
	Returns a list of whole stop words.
	"""

    sw_file = open(os.path.join(cwd, "sw.txt"), "r", encoding="utf8")
    data = sw_file.read()

    sw = data.split("\n")
    sw_file.close()
    return sw


def add_words(words):
    """
	Used to add stop words to the sw list.

	Parameters
	----------
	words : list
		the new words to append to your current list.
	"""
    if not isinstance(words, list):
        raise TypeError(f"expected a list, got {type(words)}")
    backup = os.path.join(cwd, "sw_backup.txt")
    if not os.path.isfile(backup):
        shutil.copy(os.path.join(cwd, "sw.txt"), backup)

    sw = set(get_sw())
    with open(os.path.join(cwd, "sw.txt"), "a", encoding="utf8") as myfile:
        for word in words:
            if word not in sw:
                myfile.write("\n" + word)


def reset_sw():
    """
	Reset your sw list to default.	
	"""
    default_sw = open(os.path.join(cwd, "sw_backup.txt"), "r", encoding="utf8")
    with open(os.path.join(cwd, "sw.txt"), "w", encoding="utf8") as myfile:
        myfile.write(default_sw.read())
    default_sw.close()


def clear_sw(clear_backup=False):
    """
	Clear the whole list of stop words.

	Parameters
	----------
	clear_backup : bool, default False
		Whether to clear the backed up list or not.
		Please note that if you cleared the backed up list,
		you won't be able to reset to the default sw again.
	"""
    backup = os.path.join(cwd, "sw_backup.txt")
    if not os.path.isfile(backup) and not clear_backup:
        shutil.copy(os.path.join(cwd, "sw.txt"), backup)
    elif clear_backup and os.path.isfile(backup):

        with open(backup, "w", encoding="utf8") as myfile:
            myfile.write("")

    with open(os.path.join(cwd, "sw.txt"), "w", encoding="utf8") as myfile:
        myfile.write("")
