import datapackage
import pandas as pd

data_url = 'https://datahub.io/core/finance-vix/datapackage.json'

# to load Data Package into storage
package = datapackage.Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        print (resource.descriptor['path'])
