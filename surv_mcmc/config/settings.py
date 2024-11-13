from db_utils.os_utils import get_project_root_path

root = get_project_root_path()


class empire_analytics_db:
    host = "dev-analytics-proxy.proxy-cttxmi29ul03.us-west-1.rds.amazonaws.com"
    sql_username = "empireanalytics"
    sql_password = "V4xrGHj$904N"
    sql_port = 3306
    ssh_port = 22
    ssh_host = "empiretest2.net"
    ssh_user = "ubuntu"
    default_db = "empire_analytics"
    key = str(root) + "\config\empireweb.pem"
    pem_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA6R+k5wOEGs3ILdeCwGXMCp5MrXJC3DTPb3jQkxCLVRWoqdE79wfGJ+wUhoy/
cgQmKTFLgymDaDNX8yH2ZbDBUfsCBASX9pRKSQMvvFAawnRsRXZOHLuQDOWfZPmyzQ3UP4EZJHoN
zySRXM+B+61t6a07YYSRfA3JAszE89aWPUqFeVU/XPa/grOHqkrcudBMbpR2IXuMdWqETD9vDoVO
Ogekn+ibBpG/ccICq3KpZLGuo46Rkba9T6KcFGUCW9sDpGCOumPKB30FZi85sGBHbfvClsX0qsxF
idviyDOP2AMMavwz8S8SWb15fIBKnBAGebGf4mqWDrgAE6hJfRSu/wIDAQABAoIBAQCuJXptLD46
O7EnNY+yJBlZcUl8ZBe4Iic3cXv18Gz1QXm+adQuxHrthbkLjgbRqHHNYcq0D4XfiENSF/PVoUW4
9RZbZAcVJ0+a3SuCtCBZVNkwxqCxbBe+2qXIq5M7BPKiAdGDYz3/mKSSPV0vYi3yWvs57Os1TaL0
WX1jXdDbtOt9PUYEYpIl4d1if5E6cNs3iZKyojHfJ1GWptw4YyHYQ/zou4CrIS18CUJlcp0XZS6u
3e+3Gz6vOo2NcV3CpmwgYffSuGe9x1ES/oDmdovaY0P/My9/yH1M0WEfOBrtje93dAOvMNsu6ixa
72Z9pIRTDTgUTpGgAzcMjE8PJyDRAoGBAPuVBpeDjLtXedQ89PnFXg1nFZx4jyrxIeeLaCvXtsaT
9w1ly1r7H0DZnT/sbrAJY0iUMuR/53CEgQMqJ9tlqE1pJPeZeuBtNDNUrM8QJPRDZ/Bz/Qy3Mb0R
qwuQp6lUzDYwfCBLD4v808s6CSNfH1UhINgcwFmERv7fmf+Hy72ZAoGBAO03o5seRdVgbmM/a1EV
nn3BV+HE4kE5sJlgHcL8PpUkkqG70xrfhMBRRnW6hLoRr0L/ru4ujIqanAAwd0BdTzDJS79Da6hx
Ma1tNsieZ4pRWF6TdxqbF1HC4nZcYQFq/muRp/P3APLPXBtyjFApHItCdZqp0RbB9Zp1WswvmUBX
AoGBANy9ypXgn/e07jlx7sTFDxwVadnr0jqsbsI6CIIHeODN7UQ9H+vJUYfWKOpwqkIdSpbhKbJi
I5EPQh+jumr/zGd3rS7u5OjMCLRNRH8+PB5ykl1heBPTHXo2aWzxVJr3w2J8sjynf5GlmbPETUZS
7SwFsErPF1qz/JMfYIDcBsFpAoGAG5II6OF22lrv2YcocO8jUZhkH3Bjn5MV7G2YZ+4rU+hBRpzu
50OtSTDpEIvSG1Is3YucCEDOwtk/YmI/qfJuXsw2io9KdvRZAsAbyutmO1RDAhL/l88Iwc71xS3t
dF28HkJ6k8dUi+P/34zLtBFjKOhbhNNR4uQ+KGqVV//8PtcCgYEAo7MJS+WtjNcgu5qsOCALqSr8
jWsYvDmW81cxT51XR5VYdT028R2cd7OuWUdRN4Qu+hcZKDG4o93Znw3gfFupn/ndpl0qXVGrVtjt
BTSDTRMfhaf23Q3R3wsyeH3lEd+LD33K6N0y9POV7gve3jdWXKLUbiA9bvWEpjwJ53Q3Mo0=
-----END RSA PRIVATE KEY-----"""


class empire_backstage_db:
    host = "dev-empiredistribution-rds-proxy.proxy-cttxmi29ul03.us-west-1.rds.amazonaws.com"
    sql_username = "empiredistribution"
    sql_password = "7dYR5ty20B!R"
    sql_port = 3306
    ssh_port = 22
    ssh_host = "empiretest2.net"
    ssh_user = "ubuntu"
    default_db = "empiredistribution"


class empire_snowflake_db_staging:
    user = "KONSTANTIN"
    passw = "#8Y*}NRrDC?CH.d1"
    account = "VMEIDCD-WJB65740"
    warehouse = "COMPUTE_WH"
    database = "PROD_DWH"
    schema = "STAGING"
