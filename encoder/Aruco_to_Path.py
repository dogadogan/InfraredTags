import re

"""
Website for generating ArUco markers as SVG (to be used as input):
https://chev.me/arucogen/

"""


##MAIN CONVERSION FUNCTION
def convertRect_to_PathSVG(file_in, file_out):
    """
    :param file_in: input file_path
    :param file_out: output file_path
    :return: Nothing, creates new svg file at file_out
    """
    infile, outfile = open(file_in), open(file_out, "w")
    text = infile.read()
    outfile.write("<svg xmlns=\"http://www.w3.org/2000/svg\" shape-rendering=\"crispEdges\">")
    rect_set = makePathSet(getRects(text))
    dom = "<path d=\""
    for r in rect_set:
        dom = dom + r
    dom = dom + "\"></path>"
    outfile.write(dom)
    outfile.write("</svg>")
    outfile.close()


def getAttribute(dom, *attribute):
    """
    :param dom: <dom></dom> in str form
    :param attribute: any number of attribute elements
    :return: tuple of coressponding attribute values
    """
    result = []
    for a in attribute:
        value = re.search(a + "=\"(.*?)\"", dom).group(1)
        result.append(value)
    return result


def getRects(svg):
    """
    :param svg: text of svg
    :return: each rect dom in text form
    """
    return re.findall("(<rect.*?>)", svg, re.DOTALL)


def makePaths(x, y, width, height):
    """
    :param x:
    :param y:
    :param width:
    :param height:
    :return: path list corresponding to input rectangle
    """
    result = []
    for i in range(x, x + width):
        for j in range(y, y + height):
            result.append("M{},{}h1v1h-1z ".format(i, j))
    return result


def makePathSet(rect_dom_list):
    """
    :param rect_dom_list: list of rectangle doms
    :return: set of square paths that correspond to the rectangle doms
    """
    result = set()
    for rect in rect_dom_list:
        x, y, width, height, color = getAttribute(rect, "x", "y", "width", "height", "fill")
        paths = makePaths(int(x), int(y), int(width), int(height))
        if color == "black":
            result = result.union(set(paths))
        elif color == "white":
            result = result.difference(set(paths))

    return result


if __name__ == "__main__":
    infile = "C:/Users/Ahmad/Downloads/4x4_1000-6.svg"
    outfile = "C:/Users/Ahmad/Downloads/4x4_1000-6_new.svg"
    convertRect_to_PathSVG(infile, outfile)
