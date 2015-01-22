import barrett.util as util

def main():
	chains = ['RD.txt']

	for c in chains:
		util.convert_chain([c], headers, units, c.split('.')[0]+'.h5', 5000)

headers = [
'mult',
'-2lnL',
'D_1',
'D_2',
'D_3',
'D_4',
'D_5',
'D_6',
'log(m_{\chi})',
'm_b',
'm_t',
'\\alpha_s',
'\\alpha_{em}',
'ft^p_u',
'ft^p_d',
'ft^p_s',
'del^p_u',
'del^p_d',
'del^p_s',
'\\rho_l',
'\sigma_v',
'v_{esc}',
'v_{LSR}',
'\Omega_{\chi}h^2',
'\sigma_p^{SI}',
'\sigma_p^{SD}',
'\sigma_n^{SD}',
'R',
'<\sigma v>']

units = [
'',
'',
'GeV^{-2}',
'GeV^{-2}',
'GeV^{-2}',
'GeV^{-2}',
'GeV^{-2}',
'GeV^{-2}',
'GeV',
'GeV',
'GeV',
'',
'',
'',
'',
'',
'',
'',
'',
'GeV/cm^3',
'Km/s',
'Km/s',
'Km/s',
'',
'pb',
'pb',
'pb',
'counts/kg/day/keV',
'cm^3 s^{-1}']

if __name__ == '__main__':
	main()