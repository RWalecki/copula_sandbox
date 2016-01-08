import Multivariate_Sampling as ms


mvd = ms.MVD(type='frank',para=[-10])
mvd.add_margin('uniform',para=[0,1])
mvd.add_margin('sigmoid',para=[0,1])
X = mvd.generate_x(300)

mvd = ms.MVD(type='frank')
mvd.add_margin('uniform')
mvd.add_margin('sigmoid')
mvd.fit(X)

mvd.visual_model('/tmp/tmp.pdf',samples = X)

'''
X = mvd.generate_x()

mvd.fit(X)
print mvd.margin_para
print mvd.copula_para
print

X = mvd.generate_x(10000)
mvd.fit(X)
print mvd.margin_para
print mvd.copula_para
print

X = mvd.generate_x(10)
mvd.fit(X)
print mvd.margin_para
print mvd.copula_para
print
'''
