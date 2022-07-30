from aequitas.group import Group
import pandas as pd


def create_slices_data(cat_features, test, predictions, lb):
    cat_features.append('salary')
    slices_df = test[cat_features]
    slices_df.reset_index(inplace=True, drop=True)
    slices_df['label_value'] = lb.transform(slices_df['salary']).ravel()
    slices_df = slices_df.drop('salary', axis=1)
    preds = pd.DataFrame(predictions, columns=['score'])
    slices_df = pd.concat([slices_df, preds], axis=1)
    return slices_df


def slices_performance(slices_df):
    g = Group()
    xtab, _ = g.get_crosstabs(slices_df)
    absolute_metrics = g.list_absolute_metrics(xtab)
    check_df = xtab[['attribute_name', 'attribute_value'] +
                    absolute_metrics].round(2)
    check_df.to_csv(r'slice_model_output.txt', sep='\t')
