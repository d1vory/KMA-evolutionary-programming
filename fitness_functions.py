import re

import models


class FH(models.Function):
    def _f(self, arg):
        return float(len(re.findall("0", arg)))
